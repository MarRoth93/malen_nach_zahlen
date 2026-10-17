"""Malen nach Zahlen aus eigenen Fotos.

    python malen.py foto.jpg              # vergleicht A4/A3/A2, empfiehlt eins, fragt nach
    python malen.py foto.jpg --format A3  # ohne Vergleich direkt erzeugen
"""
import argparse
import sys
import time
from pathlib import Path

from pbn.output import comparison_png, write_outputs
from pbn.paints import build_candidates, load_palette
from pbn.pipeline import FORMATS_MM, MIN_WIDTH_FLOOR_MM, Settings, load_photo, make_plan
from pbn.recommend import summarize, recommend

HERE = Path(__file__).resolve().parent


def ask_format(rec):
    while True:
        try:
            answer = input(f"Format wählen (A4/A3/A2) [Enter = {rec}]: ").strip().upper()
        except EOFError:
            return rec
        if not answer:
            return rec
        if answer in FORMATS_MM:
            return answer
        print("Bitte A4, A3 oder A2 eingeben.")


def main():
    p = argparse.ArgumentParser(description="Malen nach Zahlen aus eigenen Fotos")
    p.add_argument("foto", type=Path)
    p.add_argument("--format", choices=list(FORMATS_MM),
                   help="ohne Angabe: alle Formate vergleichen, Empfehlung zeigen, dann fragen")
    p.add_argument("--farben", type=int, default=30, help="max. Anzahl Farben (Standard 30)")
    p.add_argument("--rezepte", type=int, default=8,
                   help="max. Farben nach Mischrezept, Aufhellungen nicht mitgezählt (Standard 8)")
    p.add_argument("--min-breite", type=float, default=2.5,
                   help="jede Fläche muss an ihrer breitesten Stelle so breit sein, in mm "
                        f"(Standard 2.5, mindestens {MIN_WIDTH_FLOOR_MM})")
    p.add_argument("--min-engstelle", type=float, default=1.5,
                   help="schmalere Spitzen und Engstellen gehen an die Nachbarflächen, in mm "
                        "(Standard 1.5)")
    p.add_argument("--palette", type=Path, default=HERE / "palettes" / "platzhalter.json")
    p.add_argument("--ausgabe", type=Path, default=HERE / "ausgabe")
    args = p.parse_args()
    if args.min_breite < MIN_WIDTH_FLOOR_MM:
        p.error(f"--min-breite muss mindestens {MIN_WIDTH_FLOOR_MM} mm sein, sonst passen "
                f"zweistellige Zahlen nicht in die Flächen")
    if not 0 < args.min_engstelle <= args.min_breite:
        p.error("--min-engstelle muss größer als 0 und höchstens so groß wie --min-breite sein")

    pal = load_palette(args.palette)
    if pal.is_placeholder:
        print("Hinweis: Platzhalter-Farben – Vorschau und Rezepte sind noch nicht verlässlich.")
    cand = build_candidates(pal)
    rgb = load_photo(args.foto)
    st = Settings(max_colors=args.farben, max_recipes=args.rezepte,
                  min_width_mm=args.min_breite, min_neck_mm=args.min_engstelle)
    title = args.foto.stem
    out_dir = args.ausgabe / title

    formats = [args.format] if args.format else list(FORMATS_MM)
    plans = {}
    for fmt in formats:
        t = time.time()
        print(f"Berechne {fmt} …", end=" ", flush=True)
        plans[fmt] = make_plan(rgb, fmt, pal, cand, st)
        print(f"{plans[fmt].n_regions} Flächen ({time.time() - t:.0f} s)")

    if args.format:
        fmt = args.format
    else:
        summaries = [summarize(plans[f], cand) for f in formats]
        rec, reasons = recommend(summaries)
        print(f"\n{'Format':<7}{'Flächen':>8}{'Zeit':>8}{'Tuben':>7}{'Aufh.':>7}{'Rezepte':>8}"
              f"{'Farbe':>8}{'Konturen':>10}{'Px/mm':>7}")
        for s in summaries:
            print(f"{s.fmt + (' ★' if s.fmt == rec else ''):<7}{s.regions:>8}{s.hours:>7.0f}h"
                  f"{s.tubes:>7}{s.tints:>7}{s.recipes:>8}{s.grams:>7.0f}g{s.contours:>10.0%}"
                  f"{s.source_ppm:>7.1f}")
        print("\nZeit: grob 1 Min. pro Fläche. Konturen: Anteil der deutlichen Kanten im Foto, "
              "die als Flächengrenze erhalten bleiben.")
        print(f"\nEmpfehlung: {rec}")
        for r in reasons:
            print(f"  – {r}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = out_dir / "formatvergleich.png"
        comparison_png([plans[f] for f in formats], summaries, rec, cand, cmp_path)
        print(f"\nVergleichsbild: {cmp_path}")
        fmt = ask_format(rec) if sys.stdin.isatty() else rec

    pdf_path, rows = write_outputs(plans[fmt], pal, cand, out_dir, title)
    print(f"\nFarbliste {fmt}:")
    for r in rows:
        print(f"  {r['nr']:>2}  {r['text']:<60} ≈ {r['need']:.1f} g")
        if r["weigh"]:
            print(f"      abwiegen: {r['weigh']}")
    print(f"\nVorlage: {pdf_path}")
    print(f"Vorschau: {out_dir / f'{fmt}_vorschau.png'}")


if __name__ == "__main__":
    main()
