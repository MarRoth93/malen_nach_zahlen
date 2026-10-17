"""Malen nach Zahlen im Browser.

    python app.py            # startet http://127.0.0.1:8765 und öffnet den Browser
"""
import argparse
import csv
import io
import json
import multiprocessing as mp
import tempfile
import threading
import urllib.request
import uuid
import webbrowser
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, UnidentifiedImageError

from pbn.output import legend_rows, preview_rgb, template_svg, write_outputs
from pbn.paints import build_candidates, load_palette
from pbn.pipeline import FORMATS_MM, MIN_WIDTH_FLOOR_MM, Settings, layout, load_photo, make_plan
from pbn.recommend import recommend, summarize

HERE = Path(__file__).resolve().parent
PALETTE_PATH = HERE / "palettes" / "platzhalter.json"
INDEX = HERE / "gui" / "index.html"
KEEP_JOBS = 2  # finished photos kept in memory; an A2 plan is ~100 MB

_worker_palette = {}


def compute_plan(rgb, fmt, palette_path, st):
    """Runs in a worker process; palette and candidates are built once per process."""
    if palette_path not in _worker_palette:
        pal = load_palette(palette_path)
        _worker_palette[palette_path] = (pal, build_candidates(pal))
    pal, cand = _worker_palette[palette_path]
    return make_plan(rgb, fmt, pal, cand, st)


class Job:
    def __init__(self, name, rgb, st):
        self.id = uuid.uuid4().hex[:12]
        self.name, self.rgb, self.st = name, rgb, st
        self.landscape = rgb.shape[1] > rgb.shape[0]
        self.plans, self.summaries, self.cache = {}, {}, {}
        self.recommendation, self.reasons, self.error = None, [], None
        self.lock = threading.Lock()

    def status(self):
        formats = []
        for fmt in FORMATS_MM:
            page, area = layout(fmt, self.landscape)
            s = self.summaries.get(fmt)
            formats.append({"fmt": fmt, "page_mm": page, "area_mm": area, "done": s is not None,
                            "summary": s and {k: getattr(s, k) for k in (
                                "regions", "hours", "tubes", "tints", "recipes", "grams",
                                "contours", "source_ppm")}})
        return {"id": self.id, "name": self.name, "placeholder": PAL.is_placeholder,
                "formats": formats, "recommendation": self.recommendation,
                "reasons": self.reasons, "error": self.error}

    def cached(self, key, make):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = make()
            return self.cache[key]


def run_job(job):
    try:
        futures = {POOL.submit(compute_plan, job.rgb, fmt, str(PALETTE_PATH), job.st): fmt
                   for fmt in FORMATS_MM}
        for fut in as_completed(futures):
            fmt = futures[fut]
            plan = fut.result()
            job.plans[fmt] = plan
            job.summaries[fmt] = summarize(plan, CAND)
        job.recommendation, job.reasons = recommend([job.summaries[f] for f in FORMATS_MM])
    except Exception as e:  # shown in the page instead of a silent hang
        job.error = f"Berechnung fehlgeschlagen: {e}"


def export_zip(job, fmt):
    plan = job.plans[fmt]
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        tmp = Path(tmp)
        _, rows = write_outputs(plan, PAL, CAND, tmp, job.name)
        (tmp / f"{fmt}_vorlage.svg").write_text(template_svg(plan), encoding="utf-8")
        # utf-8-sig and decimal commas so German Excel opens it as-is
        with open(tmp / f"{fmt}_farben.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Nr", "Farbe (sRGB)", "Mischung", "Abwiegen", "Bedarf (g)"])
            for r in rows:
                w.writerow([r["nr"], hex_color(r["srgb"]), r["text"], r["weigh"], f"{r['need']:.1f}".replace(".", ",")])
        for p in sorted(tmp.iterdir()):
            z.write(p, f"{job.name}_{fmt}/{p.name}")
    return buf.getvalue()


def hex_color(srgb01):
    return "#" + "".join(f"{round(c * 255):02x}" for c in srgb01)


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=False)
    return buf.getvalue()


def int_param(q, name, default, lo, hi):
    v = int(q.get(name, [default])[0])
    if not lo <= v <= hi:
        raise ValueError(f"{name} muss zwischen {lo} und {hi} liegen")
    return v


def settings_from_query(q):
    st = Settings(max_colors=int_param(q, "farben", 30, 2, 48),
                  max_recipes=int_param(q, "rezepte", 8, 0, 30),
                  min_width_mm=float(q.get("min_breite", [2.5])[0]),
                  min_neck_mm=float(q.get("min_engstelle", [1.5])[0]))
    if st.min_width_mm < MIN_WIDTH_FLOOR_MM:
        raise ValueError(f"Die Mindestbreite muss mindestens {MIN_WIDTH_FLOOR_MM} mm sein, "
                         "sonst passen zweistellige Zahlen nicht in die Flächen.")
    if not 0 < st.min_neck_mm <= st.min_width_mm:
        raise ValueError("Engstellen müssen größer als 0 und höchstens so breit wie die "
                         "Mindestbreite sein.")
    return st


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def error(self, code, msg):
        self.send_json({"error": msg}, code)

    def do_GET(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if not parts:
            return self.send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        if parts[:2] != ["api", "jobs"] or len(parts) < 3:
            return self.error(404, "Nicht gefunden")
        job = JOBS.get(parts[2])
        if job is None:
            return self.error(404, "Dieses Foto ist nicht mehr geladen. Bitte erneut hochladen.")
        if len(parts) == 3:
            return self.send_json(job.status())
        if parts[3] == "photo.jpg":
            return self.send(200, job.cached("photo", lambda: thumb_jpeg(job.rgb)), "image/jpeg")
        fmt, what = parts[3], parts[4] if len(parts) > 4 else ""
        plan = job.plans.get(fmt)
        if plan is None:
            return self.error(409, f"{fmt} ist noch nicht fertig berechnet.")
        if what == "preview.png":
            body = job.cached((fmt, "png"), lambda: png_bytes(preview_rgb(plan, CAND)))
            return self.send(200, body, "image/png")
        if what == "template.svg":
            body = job.cached((fmt, "svg"), lambda: template_svg(plan).encode())
            return self.send(200, body, "image/svg+xml")
        if what == "colors":
            rows = job.cached((fmt, "rows"), lambda: legend_rows(plan, PAL, CAND))
            return self.send_json([{**r, "srgb": hex_color(r["srgb"])} for r in rows])
        if what == "export.zip":
            body = export_zip(job, fmt)
            name = f"{job.name}_{fmt}.zip".encode("ascii", "replace").decode()
            return self.send(200, body, "application/zip",
                             {"Content-Disposition": f'attachment; filename="{name}"'})
        return self.error(404, "Nicht gefunden")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/quit":
            self.send_json({"ok": True})
            return threading.Thread(target=self.server.shutdown, daemon=True).start()
        if url.path != "/api/jobs":
            return self.error(404, "Nicht gefunden")
        try:
            st = settings_from_query(parse_qs(url.query))
        except ValueError as e:
            return self.error(400, str(e))
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            rgb = load_photo(io.BytesIO(body))
        except (UnidentifiedImageError, OSError):
            return self.error(400, "Die Datei ist kein lesbares Bild. Bitte JPG, PNG oder TIFF wählen.")
        name = Path(unquote(self.headers.get("X-Filename", "foto"))).stem or "foto"
        job = Job(name, rgb, st)
        JOBS[job.id] = job
        for old in list(JOBS)[:-KEEP_JOBS]:
            JOBS.pop(old, None)
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self.send_json({"id": job.id})


def thumb_jpeg(rgb):
    im = Image.fromarray(rgb)
    im.thumbnail((1200, 1200))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def already_running(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return b"Malen nach Zahlen" in r.read()
    except OSError:
        return False


def main():
    global PAL, CAND, POOL, JOBS
    p = argparse.ArgumentParser(description="Malen nach Zahlen im Browser")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--kein-browser", action="store_true", help="Browser nicht automatisch öffnen")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if already_running(url):  # e.g. the icon clicked a second time
        webbrowser.open(url)
        return
    PAL = load_palette(PALETTE_PATH)
    CAND = build_candidates(PAL)
    POOL = ProcessPoolExecutor(max_workers=len(FORMATS_MM), mp_context=mp.get_context("spawn"))
    JOBS = {}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Malen nach Zahlen läuft auf {url}  (Beenden: Knopf im Browser oder Strg+C)")
    if not args.kein_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        POOL.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
