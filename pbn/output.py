"""Print-ready PDF (outlines, legend, preview), preview PNG and format comparison."""
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import measure  # noqa: E402

from .paints import amounts, describe  # noqa: E402
from .pipeline import PX_PER_MM  # noqa: E402

LINE_GRAY = 0.62
NUMBER_GRAY = 0.5
LINE_PT = 0.4
ROWS_PER_PAGE = 40
plt.rcParams["pdf.fonttype"] = 42

PLACEHOLDER_NOTE = "PLATZHALTER-FARBEN: Vorschau und Rezepte erst nach der Kalibrierung verlässlich"


def preview_rgb(plan, cand):
    return np.round(cand.srgb[plan.colors][plan.labels]).astype(np.uint8)


def legend_rows(plan, pal, cand):
    area_cm2 = np.bincount(plan.labels.ravel(), minlength=len(plan.colors)) / PX_PER_MM ** 2 / 100
    rows = []
    for k, c in enumerate(plan.colors):
        recipe = cand.recipes[c]
        need, grams = amounts(recipe, area_cm2[k])
        weigh = ""
        if grams:
            weigh = " + ".join(f"{g:.2f} g" for g in grams) + f"  (Ansatz {sum(grams):.1f} g)"
        rows.append(dict(nr=k + 1, srgb=cand.srgb[c] / 255,
                         text=describe(pal, cand.kind[c], recipe), weigh=weigh, need=need))
    return rows


def outlines_mm(plan):
    ox, oy = plan.area_mm[:2]
    segs = []
    for rid, sl in enumerate(ndi.find_objects(plan.regions), start=1):
        if sl is None:
            continue
        mask = np.pad(plan.regions[sl] == rid, 1).astype(np.uint8)
        for c in measure.find_contours(mask, 0.5):
            c = measure.approximate_polygon(c, tolerance=0.35)
            segs.append(np.column_stack([(c[:, 1] + sl[1].start - 0.5) / PX_PER_MM + ox,
                                         (c[:, 0] + sl[0].start - 0.5) / PX_PER_MM + oy]))
    return segs


def page(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)
    ax.axis("off")
    return fig, ax


def template_page(plan, title, placeholder):
    fig, ax = page(*plan.page_mm)
    ax.add_collection(LineCollection(outlines_mm(plan), linewidths=LINE_PT,
                                     colors=[(LINE_GRAY,) * 3], joinstyle="round"))
    ox, oy = plan.area_mm[:2]
    for rid in range(1, plan.n_regions + 1):
        r_mm = plan.region_radius[rid] / PX_PER_MM
        row, col = plan.region_pos[rid]
        ax.text((col + 0.5) / PX_PER_MM + ox, (row + 0.5) / PX_PER_MM + oy,
                str(plan.region_color[rid] + 1), fontsize=float(np.clip(4 * r_mm, 4.5, 8)),
                ha="center", va="center", color=(NUMBER_GRAY,) * 3)
    footer = f"{title} · {plan.fmt} · {len(plan.colors)} Farben · {plan.n_regions} Flächen"
    if placeholder:
        footer += " · " + PLACEHOLDER_NOTE
    ax.text(ox, plan.page_mm[1] - 4, footer, fontsize=6, color=(0.55,) * 3, va="bottom")
    return fig


def legend_pages(rows, title, fmt, placeholder):
    for start in range(0, len(rows), ROWS_PER_PAGE):
        fig, ax = page(210, 297)
        ax.text(15, 18, f"Farbliste – {title} ({fmt})", fontsize=13, weight="bold")
        info = "Mengen inkl. Reserve. T. = Gewichtsteile. Testtupfer trocknen lassen, dann beurteilen."
        ax.text(15, 24, info, fontsize=7, color="0.35")
        if placeholder:
            ax.text(15, 28.5, PLACEHOLDER_NOTE, fontsize=7, color="#b03030")
        for i, r in enumerate(rows[start:start + ROWS_PER_PAGE]):
            y = 36 + i * 6.3
            ax.add_patch(Rectangle((15, y - 2.4), 9, 4.8, facecolor=r["srgb"], edgecolor="0.6",
                                   linewidth=0.3))
            ax.text(27, y, f"{r['nr']:>2}", fontsize=9, weight="bold", va="center")
            ax.text(34, y - 0.9 if r["weigh"] else y, r["text"], fontsize=7.5, va="center")
            if r["weigh"]:
                ax.text(34, y + 1.7, "abwiegen: " + r["weigh"], fontsize=6, va="center",
                        color="0.35")
            ax.text(195, y, f"≈ {r['need']:.1f} g", fontsize=7.5, va="center", ha="right")
        yield fig


def preview_page(img, title):
    fig, ax = page(210, 297)
    h, w = img.shape[:2]
    s = min(180 / w, 250 / h)
    x0, y0 = (210 - w * s) / 2, 25
    ax.imshow(img, extent=(x0, x0 + w * s, y0 + h * s, y0), interpolation="lanczos")
    ax.text(15, 15, f"Vorschau – {title}", fontsize=13, weight="bold")
    return fig


def write_outputs(plan, pal, cand, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = legend_rows(plan, pal, cand)
    img = preview_rgb(plan, cand)
    Image.fromarray(img).save(out_dir / f"{plan.fmt}_vorschau.png")
    pdf_path = out_dir / f"{plan.fmt}_vorlage.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in [template_page(plan, title, pal.is_placeholder),
                    *legend_pages(rows, title, plan.fmt, pal.is_placeholder),
                    preview_page(img, title)]:
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path, rows


def comparison_png(plans, summaries, rec, cand, path, height=900):
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 26)
    bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 30)
    tiles = []
    for plan in plans:
        im = Image.fromarray(preview_rgb(plan, cand))
        tiles.append(im.resize((round(im.width * height / im.height), height), Image.LANCZOS))
    gap, caption = 30, 190
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) + 1),
                               height + caption + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    x = gap
    for t, s in zip(tiles, summaries):
        canvas.paste(t, (x, gap))
        y = gap + height + 15
        draw.text((x, y), s.fmt + ("  ★ Empfehlung" if s.fmt == rec else ""), font=bold,
                  fill="#b03030" if s.fmt == rec else "black")
        for line in [f"{s.regions} Flächen · ca. {s.hours:.0f} h",
                     f"{s.tubes} Tuben · {s.tints} Aufhellungen · {s.recipes} Rezepte",
                     f"Konturen erhalten: {s.contours:.0%}",
                     f"Foto: {s.source_ppm:.1f} Pixel/mm"]:
            y += 36
            draw.text((x, y), line, font=font, fill="black")
        x += t.width + gap
    canvas.save(path)
