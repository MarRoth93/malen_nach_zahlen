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
from skimage import measure  # noqa: E402

from .paints import amounts, describe  # noqa: E402
from .pipeline import PX_PER_MM  # noqa: E402

LINE_GRAY = 0.62
NUMBER_GRAY = 0.5
LINE_PT = 0.4
# Printed borders are smoothed along their length (about this many px, 5 px = 1 mm) but stay
# within OUTLINE_MAX_SHIFT_PX of the pixel border, so every area keeps its size and number.
OUTLINE_SIGMA_PX = 2.5
OUTLINE_MAX_SHIFT_PX = 1.0
OUTLINE_SIMPLIFY_PX = 0.1
ROWS_PER_PAGE = 40
plt.rcParams["pdf.fonttype"] = 42

PLACEHOLDER_NOTE = "PLATZHALTER-FARBEN: Vorschau und Rezepte erst nach der Kalibrierung verlässlich"


def preview_rgb(plan, cand):
    img = np.round(cand.srgb[plan.colors][plan.labels]).astype(np.uint8)
    img[~plan.painted] = 255  # unpainted background: the white canvas
    return img


def legend_rows(plan, pal, cand, numbering):
    """numbering: (pot number per color, already in a pot per color), from Pots.assign."""
    numbers, in_pot = numbering
    area_cm2 = np.bincount(plan.labels[plan.painted], minlength=len(plan.colors)) / PX_PER_MM ** 2 / 100
    rows = []
    for k in plan.painted_colors:
        c = plan.colors[k]
        recipe = cand.recipes[c]
        need, grams = amounts(recipe, area_cm2[k])
        weigh = weigh_text(grams)
        rows.append(dict(nr=numbers[k], in_pot=in_pot[k], srgb=cand.srgb[c] / 255,
                         text=describe(pal, cand.kind[c], recipe), weigh=weigh, need=need))
    return sorted(rows, key=lambda r: r["nr"])


def weigh_text(grams):
    """Grams of each part, in the order of the recipe text, and the batch."""
    if not grams:
        return ""
    return " + ".join(f"{g:.2f} g" for g in grams) + f"  (Ansatz {sum(grams):.1f} g)"


def crack_chains(regions):
    """Every border between two areas once, on the grid of pixel edges: polylines of corner
    points (x = column, y = row; pixel (i, j) spans [j, j+1] x [i, i+1]), cut at junctions
    (corners where 3 or more borders meet) and at the image edge, plus closed loops.
    Returns [(points, closed)]. Borders along the image edge are left out (see outlines_mm)."""
    H, W = regions.shape
    hor = np.zeros((H + 1, W), bool)  # edge from corner (i, j) to (i, j + 1)
    hor[1:H] = regions[1:] != regions[:-1]
    ver = np.zeros((H, W + 1), bool)  # edge from corner (i, j) to (i + 1, j)
    ver[:, 1:W] = regions[:, 1:] != regions[:, :-1]
    # image-edge edges count for the degree, so a border reaching the edge ends there
    hor_all, ver_all = hor.copy(), ver.copy()
    hor_all[[0, H]] = True
    ver_all[:, [0, W]] = True
    deg = np.zeros((H + 1, W + 1), np.int8)
    deg[:, :-1] += hor_all
    deg[:, 1:] += hor_all
    deg[:-1, :] += ver_all
    deg[1:, :] += ver_all
    junction = deg >= 3
    done_h, done_v = ~hor, ~ver  # True: no border there, or already traced

    def next_edge(i, j):
        if j < W and not done_h[i, j]:
            done_h[i, j] = True
            return i, j + 1
        if j > 0 and not done_h[i, j - 1]:
            done_h[i, j - 1] = True
            return i, j - 1
        if i < H and not done_v[i, j]:
            done_v[i, j] = True
            return i + 1, j
        if i > 0 and not done_v[i - 1, j]:
            done_v[i - 1, j] = True
            return i - 1, j
        return None

    def walk(i, j):
        pts = [(j, i)]
        while (step := next_edge(i, j)) is not None:
            i, j = step
            pts.append((j, i))
            if junction[i, j] or i in (0, H) or j in (0, W):
                break
        return pts

    chains = []
    rows, cols = np.arange(H + 1)[:, None], np.arange(W + 1)[None]
    on_edge = (rows == 0) | (rows == H) | (cols == 0) | (cols == W)
    for i, j in np.argwhere(junction | ((deg >= 1) & on_edge)):
        while True:
            pts = walk(i, j)
            if len(pts) == 1:
                break
            chains.append((np.array(pts, float), False))
    # what is left are closed loops without junctions (e.g. an island inside one area)
    for done in (done_h, done_v):
        for i, j in np.argwhere(~done):
            if done[i, j]:
                continue
            pts = walk(i, j)
            chains.append((np.array(pts[:-1] if pts[-1] == pts[0] else pts, float), True))
    return chains


def smooth_chain(pts, closed, sigma_px=OUTLINE_SIGMA_PX, max_shift=OUTLINE_MAX_SHIFT_PX):
    """Smooth a border with repeated [1/4, 1/2, 1/4] averaging (endpoints fixed, so neighbours
    still meet at junctions); no point moves farther than max_shift px from its pixel corner."""
    p, start = pts.copy(), pts.copy()
    if len(p) < 3:
        return p
    for _ in range(int(round(sigma_px ** 2 / 0.5))):
        if closed:
            p = p + 0.5 * (0.5 * (np.roll(p, 1, 0) + np.roll(p, -1, 0)) - p)
        else:
            p[1:-1] = p[1:-1] + 0.5 * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        d = p - start
        r = np.hypot(d[:, 0], d[:, 1])
        far = r > max_shift
        p[far] = start[far] + d[far] * (max_shift / r[far])[:, None]
    return p


def outlines_mm(plan):
    """The printed lines in mm on the page: every border between two areas once, smoothed,
    so neighbouring areas share one line and close exactly; plus the frame of the painted area."""
    ox, oy = plan.area_mm[:2]
    H, W = plan.regions.shape
    lines = []
    for pts, closed in crack_chains(plan.regions):
        p = smooth_chain(pts, closed)
        lines.append(measure.approximate_polygon(np.vstack([p, p[:1]]) if closed else p,
                                                 OUTLINE_SIMPLIFY_PX))
    lines.append(np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]], float))
    return [line / PX_PER_MM + (ox, oy) for line in lines]


def number_labels(plan, numbers):
    """(x mm, y mm, text, font size pt) for each area's number (its color's pot), on the page."""
    ox, oy = plan.area_mm[:2]
    for rid in np.flatnonzero(~plan.blank[1:]) + 1:  # unpainted areas get no number
        row, col = plan.region_pos[rid]
        r_mm = plan.region_radius[rid] / PX_PER_MM
        yield ((col + 0.5) / PX_PER_MM + ox, (row + 0.5) / PX_PER_MM + oy,
               str(numbers[plan.region_color[rid]]), float(np.clip(4 * r_mm, 4.5, 8)))


def template_svg(plan, numbers):
    """The outline page as SVG in mm, for viewing on screen (same lines and numbers as the PDF)."""
    w, h = plan.page_mm
    gray = round(LINE_GRAY * 255)
    paths = " ".join("M" + " ".join(f"{x:.2f},{y:.2f}" for x, y in seg) for seg in outlines_mm(plan))
    texts = "".join(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{pt * 25.4 / 72:.2f}">{t}</text>'
                    for x, y, t, pt in number_labels(plan, numbers))
    num = round(NUMBER_GRAY * 255)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}mm" height="{h}mm"><rect width="{w}" height="{h}" fill="#fff"/>'
            f'<path d="{paths}" fill="none" stroke="rgb({gray},{gray},{gray})" '
            f'stroke-width="{LINE_PT * 25.4 / 72:.3f}" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<g font-family="DejaVu Sans, sans-serif" fill="rgb({num},{num},{num})" '
            f'text-anchor="middle" dominant-baseline="central">{texts}</g></svg>')


def page(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)
    ax.axis("off")
    return fig, ax


def template_page(plan, title, placeholder, numbers):
    fig, ax = page(*plan.page_mm)
    ax.add_collection(LineCollection(outlines_mm(plan), linewidths=LINE_PT,
                                     colors=[(LINE_GRAY,) * 3], joinstyle="round", capstyle="round"))
    ox, oy = plan.area_mm[:2]
    for x, y, text, size_pt in number_labels(plan, numbers):
        ax.text(x, y, text, fontsize=size_pt, ha="center", va="center", color=(NUMBER_GRAY,) * 3)
    footer = f"{title} · {plan.fmt} · {len(plan.painted_colors)} Farben · {plan.n_painted} Flächen"
    if placeholder:
        footer += " · " + PLACEHOLDER_NOTE
    ax.text(ox, plan.page_mm[1] - 4, footer, fontsize=6, color=(0.55,) * 3, va="bottom")
    return fig


def legend_pages(rows, title, fmt, placeholder):
    for start in range(0, len(rows), ROWS_PER_PAGE):
        fig, ax = page(210, 297)
        ax.text(15, 18, f"Farbliste – {title} ({fmt})", fontsize=13, weight="bold")
        info = ("Mengen inkl. Reserve. T. = Gewichtsteile. ★ = neue Farbe: anmischen und in einen "
                "neuen Topf füllen. Testtupfer trocknen lassen, dann beurteilen.")
        ax.text(15, 24, info, fontsize=7, color="0.35")
        if placeholder:
            ax.text(15, 28.5, PLACEHOLDER_NOTE, fontsize=7, color="#b03030")
        for i, r in enumerate(rows[start:start + ROWS_PER_PAGE]):
            y = 36 + i * 6.3
            ax.add_patch(Rectangle((15, y - 2.4), 9, 4.8, facecolor=r["srgb"], edgecolor="0.6",
                                   linewidth=0.3))
            ax.text(27, y, f"{r['nr']:>2}", fontsize=9, weight="bold", va="center")
            if not r["in_pot"]:
                ax.text(32, y, "★", fontsize=7, va="center", color="#2b3fb8")
            sub = "schon im Topf" if r["in_pot"] else "abwiegen: " + r["weigh"] if r["weigh"] else ""
            ax.text(35, y - 0.9 if sub else y, r["text"], fontsize=7.5, va="center")
            if sub:
                ax.text(35, y + 1.7, sub, fontsize=6, va="center", color="0.35")
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


def write_outputs(plan, pal, cand, out_dir, title, numbering):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = legend_rows(plan, pal, cand, numbering)
    img = preview_rgb(plan, cand)
    Image.fromarray(img).save(out_dir / f"{plan.fmt}_vorschau.png")
    template = template_page(plan, title, pal.is_placeholder, numbering[0])
    with PdfPages(out_dir / f"{plan.fmt}_nur_vorlage.pdf") as pdf:  # just the outlines, to print
        pdf.savefig(template)
    pdf_path = out_dir / f"{plan.fmt}_vorlage.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in [template, *legend_pages(rows, title, plan.fmt, pal.is_placeholder),
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
