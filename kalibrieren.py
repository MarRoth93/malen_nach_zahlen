"""Farben vom fotografierten Ausstreichblatt messen und als Palette speichern.

    python kalibrieren.py ausstriche/                       # alle RAW-Fotos im Ordner
    python kalibrieren.py ausstriche/ --ohne 37 39 42 47    # Felder weglassen (z. B. Metallic)

Fotos: RAW, Blatt mit allen vier Markern, Graukarte flach auf dem Feld "Graukarte hier auflegen".
Mehrere Fotos aus verschiedenen Blickwinkeln helfen gegen Glanz auf der Farbe.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy
from skimage.color import rgb2lab

from pbn.paints import linear_to_srgb

HERE = Path(__file__).resolve().parent
LAYOUT = json.loads((HERE / "Ausstreichblatt_layout.json").read_text())
RAW_SUFFIXES = {".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2"}

PX = 5  # px per mm of the straightened sheet
MARGIN = 40  # mm of canvas around the sheet; cards may overhang the bottom edge
# JJC GC-2: JJC gives no value for this card; retailers (and JJC for the GC-1) say 18 %.
GRAY_CARD = 0.18
# Sample the inner half of each field: paint overlaps the printed box, and the photos are not
# corrected for lens distortion (about 1.5 mm at the sheet corners).
INSET = 0.25
# Gloss only ever adds light. Readings more than this much brighter than a field's darkest
# reading are taken as reflections and dropped.
SHEEN_TOL = 0.06
TILE_MM = 4  # paper-white tiles for the light falloff across the sheet
CARD_ZONE = (40, 246, 145, 100)  # x, y, w, h mm: where cards lie; no paper tiles from there


def develop(path):
    """Linear sRGB, camera white balance, no tone curve, no auto brightness, half size."""
    with rawpy.imread(str(path)) as raw:
        im = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
                             half_size=True, output_color=rawpy.ColorSpace.sRGB, user_flip=0)
    return im.astype(np.float32) / 65535


def straighten(lin):
    """Warp the photo onto the sheet (PX px/mm, MARGIN mm around it), or None without markers."""
    view = (linear_to_srgb(lin * 0.18 / np.median(lin))).astype(np.uint8)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    s = LAYOUT["marker_size_mm"]
    src, dst = [], []
    for k, (x, y) in LAYOUT["markers_topleft_mm"].items():
        if int(k) in found:
            src += list(found[int(k)])
            dst += [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    if len(src) < 12:  # fewer than three markers
        return None
    H, _ = cv2.findHomography(np.array(src), (np.array(dst, float) + MARGIN) * PX)
    w, h = LAYOUT["page_mm"]
    return cv2.warpPerspective(lin, H, (int((w + 2 * MARGIN) * PX), int((h + 2 * MARGIN) * PX)),
                               flags=cv2.INTER_AREA)


def pixels(sheet, x, y, w, h, inset=0.0):
    f = lambda v: int(round((v + MARGIN) * PX))
    return sheet[f(y + inset * h):f(y + h - inset * h), f(x + inset * w):f(x + w - inset * w)]


def poly_terms(x, y, deg=3):
    x, y = (np.asarray(x) - 105) / 105, (np.asarray(y) - 148) / 148
    return np.stack([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def paper_field(sheet):
    """Brightness of bare paper anywhere on the sheet, per channel: a smooth fit to paper tiles."""
    r = LAYOUT["references"]["paper_white"]
    ref = pixels(sheet, r["x_mm"], r["y_mm"], r["w_mm"], r["h_mm"], 0.15).reshape(-1, 3)
    lum = ref.mean(1)
    ref = np.median(ref[lum > np.percentile(lum, 30)], 0)  # without the printed text
    rel = cv2.GaussianBlur(sheet, (0, 0), PX * 0.7) / ref  # per-pixel noise is high at high ISO
    chroma = (rel.max(2) - rel.min(2)) / np.maximum(rel.mean(2), 1e-6)
    s = LAYOUT["marker_size_mm"]
    blocked = [CARD_ZONE] + [(x - 2, y - 2, s + 4, s + 4) for x, y in LAYOUT["markers_topleft_mm"].values()]
    tiles = []
    w, h = LAYOUT["page_mm"]
    for ty in range(2, h - TILE_MM, TILE_MM):
        for tx in range(2, w - TILE_MM, TILE_MM):
            cx, cy = tx + TILE_MM / 2, ty + TILE_MM / 2
            if any(bx <= cx <= bx + bw and by <= cy <= by + bh for bx, by, bw, bh in blocked):
                continue
            if (pixels(chroma, tx, ty, TILE_MM, TILE_MM) < 0.12).mean() > 0.97:
                tiles.append((cx, cy, *np.median(pixels(sheet, tx, ty, TILE_MM, TILE_MM).reshape(-1, 3), 0)))
    tiles = np.array(tiles)
    lum = tiles[:, 2:].mean(1)
    tiles = tiles[abs(lum / np.median(lum) - 1) < 0.35]  # drops table, text and shadows
    terms = poly_terms(tiles[:, 0], tiles[:, 1])
    coef = np.stack([np.linalg.lstsq(terms, np.log(tiles[:, 2 + c]), rcond=None)[0] for c in range(3)], 1)
    return lambda x, y: np.exp(poly_terms(x, y) @ coef)


def measure(sheet):
    """Fields relative to the paper at their spot; paper reflectance from the gray card, if valid."""
    field = paper_field(sheet)
    rel = []
    for s in LAYOUT["swatches"]:
        p = pixels(sheet, s["x_mm"], s["y_mm"], s["w_mm"], s["h_mm"], INSET).reshape(-1, 3)
        rel.append(np.median(p, 0) / field(s["x_mm"] + s["w_mm"] / 2, s["y_mm"] + s["h_mm"] / 2))
    g = LAYOUT["references"]["gray_card"]
    # lower part of the field: a card laid on it rarely sits exactly on the field's top edge
    card = pixels(cv2.GaussianBlur(sheet, (0, 0), PX * 0.7),
                  g["x_mm"] + 6, g["y_mm"] + 8, g["w_mm"] - 12, g["h_mm"] - 9).reshape(-1, 3)
    card_med = np.median(card, 0)
    at_card = field(g["x_mm"] + g["w_mm"] / 2, g["y_mm"] + 12)
    lum = card.mean(1)
    # a single flat card is uniform and much darker than paper; fanned or missing cards are not
    card_ok = lum.std() / lum.mean() < 0.08 and 0.12 < (card_med / at_card).mean() < 0.45
    paper = GRAY_CARD * at_card / card_med if card_ok else None
    return np.array(rel), paper


def combine(rels):
    """Per field: drop readings brightened by gloss, average the rest."""
    rels = np.stack(rels)  # (shots, fields, 3)
    Y = rels.mean(2)
    keep = Y <= Y.min(0) * (1 + SHEEN_TOL)
    out = np.array([np.median(rels[keep[:, k], k], 0) for k in range(rels.shape[1])])
    return out, keep.sum(0)


def write_palette(path, paints, white_nr, n_shots):
    lines = [json.dumps(p, ensure_ascii=False) for p in paints]
    head = {
        "source": "calibrated",
        "note": f"Gemessen mit kalibrieren.py aus {n_shots} Fotos des Ausstreichblatts, "
                f"Graukarte = {GRAY_CARD:.0%}. Deckkraft (S) noch nicht gemessen.",
        "white_nr": white_nr,
    }
    text = json.dumps(head, ensure_ascii=False, indent=2)[:-2]
    text += ',\n  "paints": [\n    ' + ",\n    ".join(lines) + "\n  ]\n}\n"
    path.write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Farben vom Ausstreichblatt messen")
    p.add_argument("ordner", type=Path, help="Ordner mit RAW-Fotos des Ausstreichblatts")
    p.add_argument("--ohne", type=int, nargs="*", default=[], help="Felder weglassen, z. B. Metallic")
    p.add_argument("--weiss", type=int, help="Feld mit dem Weiß zum Mischen (Standard: hellstes neutrales)")
    p.add_argument("--palette", type=Path, default=HERE / "palettes" / "gemessen.json",
                   help="Ausgabe; Namen und Codes einer vorhandenen Datei bleiben erhalten")
    args = p.parse_args()

    photos = sorted(f for f in args.ordner.iterdir() if f.suffix.lower() in RAW_SUFFIXES)
    if not photos:
        p.error(f"keine RAW-Fotos in {args.ordner}")
    rels, papers = [], []
    for f in photos:
        sheet = straighten(develop(f))
        if sheet is None:
            print(f"{f.name}: Marker nicht gefunden, übersprungen")
            continue
        rel, paper = measure(sheet)
        rels.append(rel)
        if paper is not None:
            papers.append(paper)
        print(f"{f.name}: gemessen, Graukarte {'ok' if paper is not None else 'nicht flach auf ihrem Feld'}")
    if not rels:
        p.error("in keinem Foto alle Marker gefunden")
    if not papers:
        p.error("in keinem Foto liegt die Graukarte flach auf dem Feld 'Graukarte hier auflegen'")

    rel, used = combine(rels)
    lin = rel * np.median(papers, 0)
    srgb = np.round(linear_to_srgb(lin)).astype(int)
    lab = rgb2lab(srgb / 255)
    fields = [s["nr"] for s in LAYOUT["swatches"] if s["nr"] not in args.ohne]
    idx = {s["nr"]: k for k, s in enumerate(LAYOUT["swatches"])}
    if args.weiss is None:
        neutral = [n for n in fields if np.hypot(*lab[idx[n], 1:]) < 8]
        if not neutral:
            p.error("kein neutrales Weiß gefunden, bitte --weiss angeben")
        args.weiss = max(neutral, key=lambda n: lab[idx[n], 0])
    elif args.weiss not in fields:
        p.error(f"--weiss {args.weiss} ist kein gemessenes Feld")

    old = {}
    if args.palette.exists():
        old = {q["nr"]: q for q in json.loads(args.palette.read_text(encoding="utf-8"))["paints"]}
    paints = []
    print(f"\n{'Feld':>4}  {'Code':<6}{'Farbe':<9}{'L*':>5}{'Fotos':>7}")
    for n in fields:
        k = idx[n]
        prev = old.get(n, {})
        paints.append({"nr": n, "name": prev.get("name", f"Feld {n}"), "code": prev.get("code", ""),
                       "srgb": srgb[k].tolist()})
        flag = "  heller als Papier" if (lin[k] > 1).any() else ""
        print(f"{n:>4}  {prev.get('code', ''):<6}#{''.join(f'{c:02x}' for c in srgb[k])}  "
              f"{lab[k, 0]:5.1f}{used[k]:>5}/{len(rels)}{flag}")
    write_palette(args.palette, paints, args.weiss, len(rels))
    print(f"\nWeiß zum Mischen: Feld {args.weiss}. Palette: {args.palette}")


if __name__ == "__main__":
    main()
