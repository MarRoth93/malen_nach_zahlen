"""Malen nach Zahlen im Browser.

    python app.py            # startet http://127.0.0.1:8765 und öffnet den Browser
"""
import argparse
import csv
import io
import itertools
import json
import multiprocessing as mp
import tempfile
import threading
import urllib.request
import uuid
import webbrowser
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, UnidentifiedImageError

from pbn.output import legend_rows, preview_rgb, template_svg, weigh_text, write_outputs
from pbn.paints import amounts, build_candidates, load_palette
from pbn.pipeline import FORMATS_MM, MIN_WIDTH_FLOOR_MM, Settings, layout, load_photo, make_plan
from pbn.pots import MAX_POTS, Pots, pot_candidates, room_error
from pbn.recommend import recommend, summarize

HERE = Path(__file__).resolve().parent
PALETTE_PATH = HERE / "palettes" / "gemessen.json"
INDEX = HERE / "gui" / "index.html"
EXPORT_DIR = HERE / "exported"
POTS = Pots(HERE / "palettes" / "toepfe.json")
EXPORT_LOCK = threading.Lock()  # one export at a time: it hands out pot numbers, then registers them
KEEP_JOBS = 2  # finished photos kept in memory; an A2 plan is ~100 MB
MAX_VARIANTS = 2  # per format, besides the photo's own settings: three side by side
SUMMARY_KEYS = ("regions", "hours", "tubes", "tints", "recipes", "grams", "contours",
                "source_ppm", "color_error")

_worker_palette = {}


def compute_plan(rgb, fmt, palette_path, st, pot_keys):
    """Runs in a worker process; palette and candidates are built once per process.
    pot_keys: the pot list when the computation started; its colors are preferred."""
    if palette_path not in _worker_palette:
        pal = load_palette(palette_path)
        _worker_palette[palette_path] = (pal, build_candidates(pal))
    pal, cand = _worker_palette[palette_path]
    return make_plan(rgb, fmt, pal, cand, st, pot_candidates(pot_keys, pal, cand))


class Job:
    def __init__(self, name, rgb, st):
        self.id = uuid.uuid4().hex[:12]
        self.name, self.rgb, self.st = name, rgb, st
        self.landscape = rgb.shape[1] > rgb.shape[0]
        self.plans, self.summaries, self.cache = {}, {}, {}
        # Other settings for one format, id "A3-40-5": {fmt, st, summary, error}. Their plans
        # sit in self.plans under the same id, so preview, colors and export work unchanged.
        self.variants = {}
        self.recommendation, self.reasons, self.error = None, [], None
        self.lock = threading.Lock()

    def settings(self, plan_id):
        return self.variants[plan_id]["st"] if plan_id in self.variants else self.st

    def status(self):
        summary = lambda s: s and {k: getattr(s, k) for k in SUMMARY_KEYS}
        formats = []
        for fmt in FORMATS_MM:
            page, area = layout(fmt, self.landscape)
            s = self.summaries.get(fmt)
            formats.append({"fmt": fmt, "page_mm": page, "area_mm": area, "done": s is not None,
                            "summary": summary(s)})
        variants = [{"id": vid, "fmt": v["fmt"], "farben": v["st"].max_colors,
                     "rezepte": v["st"].max_recipes, "done": v["summary"] is not None,
                     "error": v["error"], "summary": summary(v["summary"])}
                    for vid, v in list(self.variants.items())]
        return {"id": self.id, "name": self.name, "placeholder": PAL.is_placeholder,
                "formats": formats, "recommendation": self.recommendation,
                "reasons": self.reasons, "error": self.error,
                "base": {"farben": self.st.max_colors, "rezepte": self.st.max_recipes},
                "variants": variants}

    def cached(self, key, make):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = make()
            return self.cache[key]


def run_job(job):
    try:
        futures = {POOL.submit(compute_plan, job.rgb, fmt, str(PALETTE_PATH), job.st, POTS.keys(in_use_only=True)): fmt
                   for fmt in FORMATS_MM}
        for fut in as_completed(futures):
            fmt = futures[fut]
            plan = fut.result()
            job.plans[fmt] = plan
            job.summaries[fmt] = summarize(plan, CAND)
        job.recommendation, job.reasons = recommend([job.summaries[f] for f in FORMATS_MM])
    except Exception as e:  # shown in the page instead of a silent hang
        job.error = f"Berechnung fehlgeschlagen: {e}"


def add_variant(job, fmt, colors, recipes):
    """Start computing fmt with other color/recipe limits; returns the plan id to show."""
    if (colors, recipes) == (job.st.max_colors, job.st.max_recipes):
        return fmt
    vid = f"{fmt}-{colors}-{recipes}"
    with job.lock:
        if vid in job.variants:
            return vid
        if sum(v["fmt"] == fmt for v in job.variants.values()) >= MAX_VARIANTS:
            raise ValueError(f"Höchstens {MAX_VARIANTS + 1} Varianten nebeneinander. "
                             "Entferne erst eine.")
        st = replace(job.st, max_colors=colors, max_recipes=recipes)
        job.variants[vid] = {"fmt": fmt, "st": st, "summary": None, "error": None}
    threading.Thread(target=run_variant, args=(job, vid, st), daemon=True).start()
    return vid


def run_variant(job, vid, st):
    fmt = job.variants[vid]["fmt"]
    try:
        plan = POOL.submit(compute_plan, job.rgb, fmt, str(PALETTE_PATH), st, POTS.keys(in_use_only=True)).result()
        summary = summarize(plan, CAND)
    except Exception as e:  # shown on the variant instead of a silent hang
        with job.lock:
            if vid in job.variants:
                job.variants[vid]["error"] = f"Berechnung fehlgeschlagen: {e}"
        return
    with job.lock:
        if vid in job.variants:  # not removed while it was computing
            job.plans[vid] = plan
            job.variants[vid]["summary"] = summary


def remove_variant(job, vid):
    with job.lock:
        job.variants.pop(vid, None)
        job.plans.pop(vid, None)
        for key in [k for k in job.cache if isinstance(k, tuple) and k[0] == vid]:
            del job.cache[key]


def export_zip(job, plan_id, stem, numbering):
    plan = job.plans[plan_id]
    fmt = plan.fmt
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        tmp = Path(tmp)
        _, rows = write_outputs(plan, PAL, CAND, tmp, job.name, numbering)
        (tmp / f"{fmt}_vorlage.svg").write_text(template_svg(plan, numbering[0]), encoding="utf-8")
        # utf-8-sig and decimal commas so German Excel opens it as-is
        with open(tmp / f"{fmt}_farben.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Nr", "Farbe (sRGB)", "Mischung", "Abwiegen", "Bedarf (g)", "Topf"])
            for r in rows:
                w.writerow([r["nr"], hex_color(r["srgb"]), r["text"], r["weigh"], f"{r['need']:.1f}".replace(".", ","),
                            "schon im Topf" if r["in_pot"] else "neu"])
        for p in sorted(tmp.iterdir()):
            z.write(p, f"{stem}/{p.name}")
    return buf.getvalue()


def save_export(job, plan_id, register=True):
    """Write the export ZIP into EXPORT_DIR (an earlier export of the same name is kept), then
    give the template's new colors their pots, unless register is False."""
    plan, st = job.plans[plan_id], job.settings(plan_id)
    stem = f"{job.name}_{plan.fmt}_{st.max_colors}-Farben_{st.max_recipes}-Rezepte"
    with EXPORT_LOCK:
        numbering = POTS.assign_plan(plan, PAL, CAND)
        msg = room_error(*numbering)
        if msg:
            raise ValueError(msg)
        body = export_zip(job, plan_id, stem, numbering)
        EXPORT_DIR.mkdir(exist_ok=True)
        for n in itertools.count(1):
            path = EXPORT_DIR / (f"{stem}.zip" if n == 1 else f"{stem}_{n}.zip")
            try:
                with open(path, "xb") as f:  # "x": never overwrite, also with two clicks at once
                    f.write(body)
                break
            except FileExistsError:
                continue
        if register:
            POTS.register_plan(plan, PAL, CAND, numbering[0], path.stem)
        return path


def pot_weigh(pot):
    """Grams to weigh for one full pot, in the order of the recipe text (white last)."""
    white = PAL.nr[PAL.white]
    parts = sorted(pot["recipe"], key=lambda field_parts: field_parts[0] == white)
    return weigh_text(amounts(parts, 0)[1])


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
                  min_width_mm=float(q.get("min_breite", [2.0])[0]),
                  min_neck_mm=float(q.get("min_engstelle", [1.0])[0]),
                  blank_background=q.get("weiss_lassen", ["0"])[0] == "1")
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
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if parts == ["api", "pots"]:
            return self.send_json({"pots": [{**p, "weigh": pot_weigh(p)} for p in POTS.all()],
                                   "max": MAX_POTS})
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
        # numbers come from the pot list, so these are cached per version of it
        if what == "template.svg":
            body = job.cached((fmt, "svg", POTS.version),
                              lambda: template_svg(plan, POTS.assign_plan(plan, PAL, CAND)[0]).encode())
            return self.send(200, body, "image/svg+xml")
        if what == "colors":
            rows = job.cached((fmt, "rows", POTS.version),
                              lambda: legend_rows(plan, PAL, CAND, POTS.assign_plan(plan, PAL, CAND)))
            return self.send_json([{**r, "srgb": hex_color(r["srgb"])} for r in rows])
        return self.error(404, "Nicht gefunden")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/quit":
            self.send_json({"ok": True})
            return threading.Thread(target=self.server.shutdown, daemon=True).start()
        parts = [p for p in url.path.split("/") if p]
        if parts == ["api", "pots", "verwenden"]:  # all at once
            POTS.set_all_in_use(parse_qs(url.query).get("an", ["1"])[0] == "1")
            return self.send_json({"ok": True})
        if len(parts) == 4 and parts[:2] == ["api", "pots"] and parts[2].isdigit() and parts[3] == "verwenden":
            on = parse_qs(url.query).get("an", ["1"])[0] == "1"
            if not POTS.set_in_use(int(parts[2]), on):
                return self.error(404, f"Topf {parts[2]} gibt es nicht.")
            return self.send_json({"ok": True})
        if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[4] in ("export", "variants"):
            job = JOBS.get(parts[2])
            if job is None:
                return self.error(404, "Dieses Foto ist nicht mehr geladen. Bitte erneut hochladen.")
            if parts[4] == "variants":
                if parts[3] not in FORMATS_MM:
                    return self.error(404, "Nicht gefunden")
                try:
                    q = parse_qs(url.query)
                    vid = add_variant(job, parts[3], int_param(q, "farben", 30, 2, 48),
                                      int_param(q, "rezepte", 8, 0, 30))
                except ValueError as e:
                    return self.error(400, str(e))
                return self.send_json({"id": vid})
            if parts[3] not in job.plans:
                return self.error(409, f"{parts[3]} ist noch nicht fertig berechnet.")
            try:
                path = save_export(job, parts[3], parse_qs(url.query).get("toepfe", ["1"])[0] != "0")
            except ValueError as e:  # no pot numbers left
                return self.error(409, str(e))
            except OSError as e:
                return self.error(500, f"Export fehlgeschlagen: {e}")
            return self.send_json({"file": str(path.relative_to(HERE))})
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

    def do_DELETE(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "pots"] and parts[2].isdigit():
            if not POTS.free(int(parts[2])):
                return self.error(404, f"Topf {parts[2]} gibt es nicht.")
            return self.send_json({"ok": True})
        if len(parts) != 4 or parts[:2] != ["api", "jobs"]:
            return self.error(404, "Nicht gefunden")
        job = JOBS.get(parts[2])
        if job is None:
            return self.error(404, "Dieses Foto ist nicht mehr geladen. Bitte erneut hochladen.")
        if parts[3] not in job.variants:
            return self.error(400, "Nur zusätzliche Varianten lassen sich entfernen.")
        remove_variant(job, parts[3])
        self.send_json({"ok": True})


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
    global PAL, CAND, POOL, JOBS, PAGE
    p = argparse.ArgumentParser(description="Malen nach Zahlen im Browser")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--kein-browser", action="store_true", help="Browser nicht automatisch öffnen")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if already_running(url):  # e.g. the icon clicked a second time
        webbrowser.open(url)
        return
    PAGE = INDEX.read_bytes()  # read once, so the page always matches the code this process runs
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
