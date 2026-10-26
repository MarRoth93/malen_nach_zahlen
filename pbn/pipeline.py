"""Photo -> numbered areas for one paper format."""
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.cluster import MiniBatchKMeans

from .paints import KIND_PENALTY, KL, MIX, NEW_POT_PENALTY, TUBE, color_distance, srgb_to_lab

FORMATS_MM = {"A4": (210, 297), "A3": (297, 420), "A2": (420, 594)}
MARGIN_MM = 10
PX_PER_MM = 5  # working resolution; outlines are traced at this resolution
MIN_WIDTH_FLOOR_MM = 2.0  # a 2-digit number at the smallest font size (4.5 pt) is ~2 mm wide


@dataclass
class Settings:
    max_colors: int = 30
    max_recipes: int = 8
    min_width_mm: float = 2.0  # every area is at least this wide at its widest point
    min_area_mm2: float = 8.0
    min_neck_mm: float = 1.0  # narrower spikes and necks go to the surrounding areas
    blank_background: bool = False  # white areas touching the picture's edge stay unpainted


@dataclass
class Plan:
    fmt: str
    page_mm: tuple  # (w, h)
    area_mm: tuple  # (x0, y0, w, h) of the painted area on the page
    source_ppm: float  # photo pixels per mm of painting
    colors: list  # candidate indices; color number = position + 1
    labels: np.ndarray  # (H, W) index into colors
    regions: np.ndarray  # (H, W) region id, 1..n
    region_color: np.ndarray  # (n + 1,) index into colors, per region id
    region_pos: np.ndarray  # (n + 1, 2) row, col of the number
    region_radius: np.ndarray  # (n + 1,) inscribed radius in px
    photo_L: np.ndarray  # (H, W) lightness of the cropped photo, for comparing formats
    # Mean color distance (as used to choose the colors) between the painted areas and the
    # smoothed photo, for comparing settings. Not for comparing formats: see contours_kept.
    color_error: float
    blank: np.ndarray  # (n + 1,) True: area left unpainted (white background): no number, no paint

    @property
    def n_regions(self):
        return len(self.region_color) - 1

    @property
    def painted(self):
        """(H, W) True where paint goes."""
        return ~self.blank[self.regions]

    @property
    def painted_colors(self):
        """Indices into colors that are painted somewhere; only these get a number and a pot."""
        return sorted(set(self.region_color[1:][~self.blank[1:]].tolist()))

    @property
    def n_painted(self):
        return int((~self.blank[1:]).sum())


def load_photo(path):
    im = Image.open(path)
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)
    if icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
        except ImageCms.PyCMSError as e:
            print(f"Warnung: Farbprofil nicht lesbar ({e}), nehme sRGB an.")
    return np.asarray(im.convert("RGB"))


def layout(fmt, landscape):
    w, h = FORMATS_MM[fmt]
    if landscape:
        w, h = h, w
    return (w, h), (MARGIN_MM, MARGIN_MM, w - 2 * MARGIN_MM, h - 2 * MARGIN_MM)


def fit_to_area(rgb, area_w, area_h):
    """Center-crop to the area's aspect ratio and resize to the working resolution."""
    H, W = rgb.shape[:2]
    target = area_w / area_h
    if W / H > target:
        cw = round(H * target)
        rgb = rgb[:, (W - cw) // 2:(W - cw) // 2 + cw]
    else:
        ch = round(W / target)
        rgb = rgb[(H - ch) // 2:(H - ch) // 2 + ch]
    source_ppm = rgb.shape[1] / area_w
    size = (round(area_w * PX_PER_MM), round(area_h * PX_PER_MM))
    interp = cv2.INTER_AREA if source_ppm >= PX_PER_MM else cv2.INTER_CUBIC
    return cv2.resize(np.ascontiguousarray(rgb), size, interpolation=interp), source_ppm


def fit_lightness(lab, lo, hi, L_dark, L_white):
    """Map photo lightness [lo, hi] onto the range the paints can reach."""
    out = lab.copy()
    t = np.clip((lab[..., 0] - lo) / max(hi - lo, 1e-6), 0, 1)
    out[..., 0] = L_dark + t * (L_white - L_dark)
    return out


def color_targets(lab, n=256, sample=120_000, seed=0):
    px = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    px = px[rng.choice(len(px), min(sample, len(px)), replace=False)]
    km = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=seed, batch_size=8192).fit(px)
    w = np.bincount(km.labels_, minlength=n).astype(float)
    keep = w > 0
    return km.cluster_centers_[keep], w[keep] / w.sum()


def candidate_pool(cand, targets, pots=()):
    """All tubes, tints and pots, plus the recipes closest to each target color."""
    mix_ids = np.flatnonzero(cand.kind == MIX)
    scale = np.array([1 / KL, 1, 1])
    _, nn = cKDTree(cand.lab[mix_ids] * scale).query(targets * scale, k=4)
    mixes = np.unique(mix_ids[nn.ravel()])
    return np.unique(np.concatenate([np.flatnonzero(cand.kind != MIX), mixes, np.asarray(pots, int)]))


def select_colors(D, w, kinds, max_colors, max_recipes, min_gain=1e-4):
    """Greedy pick + swap refinement of the color set minimizing weighted mean distance."""
    def allowed(sel):
        ok = np.ones(D.shape[1], bool)
        ok[sel] = False
        if sum(kinds[s] == MIX for s in sel) >= max_recipes:
            ok[kinds == MIX] = False
        return ok

    sel, cur = [], np.full(D.shape[0], 1e3)
    while len(sel) < max_colors:
        gain = w @ (cur[:, None] - np.minimum(cur[:, None], D))
        gain[~allowed(sel)] = -np.inf
        c = int(np.argmax(gain))
        if gain[c] <= min_gain:
            break
        sel.append(c)
        cur = np.minimum(cur, D[:, c])

    def objective(s):
        return w @ D[:, s].min(1)

    for _ in range(3):
        improved = False
        for i in range(len(sel)):
            rest = sel[:i] + sel[i + 1:]
            base = D[:, rest].min(1) if rest else np.full(D.shape[0], 1e3)
            obj = w @ np.minimum(base[:, None], D)
            obj[~allowed(rest)] = np.inf
            c = int(np.argmin(obj))
            if obj[c] < objective(sel) - 1e-6:
                sel[i], improved = c, True
        if not improved:
            break
    return sel


def assign_pixels(lab, colors_lab, chunk=50_000):
    """Nearest chosen color per pixel, computed once per distinct (rounded) Lab value."""
    q = np.round(lab).astype(np.int64)
    key = (q[..., 0] * 256 + q[..., 1] + 128) * 256 + q[..., 2] + 128
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    bins = np.stack([uniq // 65536, (uniq // 256) % 256 - 128, uniq % 256 - 128], 1).astype(float)
    best = np.concatenate([
        np.argmin(color_distance(bins[s:s + chunk, None], colors_lab[None]), 1)
        for s in range(0, len(bins), chunk)])
    return best[inv].reshape(lab.shape[:2])


def smooth_labels(labels, n, sigma):
    """Blur each color's mask and keep the strongest: removes speckle, rounds edges."""
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for k in range(n):
        m = cv2.GaussianBlur((labels == k).astype(np.float32), (0, 0), sigma)
        upd = m > best
        best[upd], out[upd] = m[upd], k
    return out


def boundary_clearance(regions):
    """Distance in px from each pixel center to the nearest area boundary or image border.
    Measured on a grid twice as fine, where boundaries lie between the pixels; distances
    between pixel centers instead let strips ~0.3 mm too narrow pass at diagonals."""
    H, W = regions.shape
    crack = np.zeros((2 * H + 1, 2 * W + 1), bool)
    crack[1:-1:2, 2:-1:2] = regions[:, 1:] != regions[:, :-1]
    crack[2:-1:2, 1:-1:2] = regions[1:] != regions[:-1]
    crack[[0, -1]] = True
    crack[:, [0, -1]] = True
    dist = cv2.distanceTransform((~crack).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist[1::2, 1::2] / 2


def merge_small(labels, lab, colors_lab, min_area_px, min_width_px, rounds=10):
    """Merge areas that are too small or too thin into their most similar neighbour,
    smallest first, so textures become patches that follow the photo's colors."""
    limits = np.geomspace(3, min_area_px, 5)
    scale = np.array([1 / KL, 1, 1])
    for i in range(rounds):
        final = i >= len(limits) - 1
        regions = measure.label(labels, connectivity=1, background=-1)
        n = regions.max()
        flat = regions.ravel()
        area = np.bincount(flat, minlength=n + 1)
        small = area < limits[min(i, len(limits) - 1)]
        if final:
            clear = boundary_clearance(regions)
            small[1:] |= 2 * ndi.maximum(clear, regions, np.arange(1, n + 1)) <= min_width_px - 1
        small[0] = False
        if not small.any():
            if final:
                break
            continue
        color = np.zeros(n + 1, int)
        color[flat] = labels.ravel()
        mean = np.stack([np.bincount(flat, lab[..., c].ravel(), n + 1) for c in range(3)], 1)
        mean /= np.maximum(area, 1)[:, None]

        a = np.concatenate([regions[:, :-1].ravel(), regions[:-1].ravel()])
        b = np.concatenate([regions[:, 1:].ravel(), regions[1:].ravel()])
        a, b = np.concatenate([a, b]), np.concatenate([b, a])
        # A small area only joins a bigger neighbour, so two areas never swap colors.
        m = small[a] & (a != b) & ((area[b] > area[a]) | ((area[b] == area[a]) & (b > a)))
        a, b = a[m], b[m]
        if not len(a):
            continue
        pair, length = np.unique(a * (n + 1) + b, return_counts=True)
        a, b = pair // (n + 1), pair % (n + 1)
        cost = (((mean[a] - colors_lab[color[b]]) * scale) ** 2).sum(1) - 1e-3 * length
        order = np.lexsort((cost, a))
        _, first = np.unique(a[order], return_index=True)
        winner = order[first]
        color[a[winner]] = color[b[winner]]
        labels = color[regions]
    return labels


def thin_parts(regions, clear, neck_px):
    """Spikes and necks narrower than neck_px: the pieces of each area that no disc of
    that diameter covers while staying inside the area. Returned as labels 1..k (0: none)."""
    r = neck_px / 2
    # Where such a disc fits, within half a pixel (the resolution is 0.2 mm).
    core = clear > r - 0.5
    dist, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    thin = ~((dist <= r) & (regions[iy, ix] == regions))
    pieces = measure.label(np.where(thin, regions, 0), connectivity=1, background=0)
    # Every corner loses a few pixels to this test too; only larger pieces count.
    keep = np.bincount(pieces.ravel()) >= neck_px ** 2 / 4
    keep[0] = False
    ids = np.zeros(len(keep), int)
    ids[keep] = np.arange(1, keep.sum() + 1)
    return ids[pieces]


def give_away(labels, regions, pieces):
    """Each thin piece takes the color of the other area it shares the longest border with."""
    pairs = []
    for here, there in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:, 1:], np.s_[:, :-1]),
                        (np.s_[:-1], np.s_[1:]), (np.s_[1:], np.s_[:-1])]:
        # Neighbours that are thin pieces themselves are skipped, so two never swap.
        m = (pieces[here] > 0) & (regions[there] != regions[here]) & (pieces[there] == 0)
        pairs.append(np.stack([pieces[here][m], regions[there][m]], 1))
    pairs = np.concatenate(pairs)
    if not len(pairs):
        return labels
    n = regions.max() + 1
    key, length = np.unique(pairs[:, 0] * n + pairs[:, 1], return_counts=True)
    piece, other = key // n, key % n
    order = np.lexsort((-length, piece))
    _, first = np.unique(piece[order], return_index=True)
    region_color = np.zeros(n, int)
    region_color[regions.ravel()] = labels.ravel()
    target = np.full(pieces.max() + 1, -1)
    target[piece[order[first]]] = region_color[other[order[first]]]
    new = target[pieces]
    change = new >= 0
    labels = labels.copy()
    labels[change] = new[change]
    return labels


def remove_small(labels, min_area_px, min_width_px, neck_px):
    """Let neighbours absorb areas that are too small or too thin to paint, and the parts
    of areas narrower than neck_px. Returns only once every area meets the size limits.
    Thin parts are handed out until that stops helping: a few tiny pieces can remain where
    two areas would trade them back and forth."""
    thin_left = np.inf  # thin pixels last round
    for _ in range(30):
        regions = measure.label(labels, connectivity=1, background=-1)
        n = regions.max()
        area = np.bincount(regions.ravel(), minlength=n + 1)
        clear = boundary_clearance(regions)
        width = 2 * ndi.maximum(clear, regions, np.arange(1, n + 1))
        small = np.zeros(n + 1, bool)
        small[1:] = (area[1:] < min_area_px) | (width <= min_width_px - 1)
        if small.any():
            idx = ndi.distance_transform_edt(small[regions], return_distances=False,
                                             return_indices=True)
            labels = labels[tuple(idx)]
            continue
        if thin_left:
            pieces = thin_parts(regions, clear, neck_px)
            n_thin = np.count_nonzero(pieces)
            if 0 < n_thin < thin_left:
                labels = give_away(labels, regions, pieces)
                thin_left = n_thin
                continue
            thin_left = 0
        return labels, regions, clear
    raise RuntimeError("Kleine Flächen ließen sich nicht vollständig entfernen")


def make_plan(rgb, fmt, pal, cand, st, pots=()):
    """pots: candidate indices of the colors already in pots (pbn.pots.pot_candidates)."""
    landscape = rgb.shape[1] > rgb.shape[0]
    page, area = layout(fmt, landscape)
    work, source_ppm = fit_to_area(rgb, area[2], area[3])

    tubes = np.flatnonzero(cand.kind == TUBE)
    L_dark, L_white = cand.lab[tubes, 0].min(), cand.lab[pal.white, 0]
    raw_lab = srgb_to_lab(work)
    lo, hi = np.percentile(raw_lab[..., 0], [0.5, 99.5])
    smooth = work
    for _ in range(2):
        smooth = cv2.bilateralFilter(smooth, 0, 18, 0.6 * PX_PER_MM)
    lab = fit_lightness(srgb_to_lab(smooth), lo, hi, L_dark, L_white)

    targets, w = color_targets(lab)
    pool = candidate_pool(cand, targets, pots)
    penalty = np.array([KIND_PENALTY[k] + NEW_POT_PENALTY for k in cand.kind[pool]])
    penalty[np.isin(pool, pots)] = 0.0
    D = color_distance(targets[:, None], cand.lab[pool][None]) + penalty
    chosen = pool[select_colors(D, w, cand.kind[pool], st.max_colors, st.max_recipes)]

    labels = assign_pixels(lab, cand.lab[chosen])
    labels = smooth_labels(labels, len(chosen), 0.4 * PX_PER_MM)
    min_area_px, min_width_px = st.min_area_mm2 * PX_PER_MM ** 2, st.min_width_mm * PX_PER_MM
    labels = merge_small(labels, lab, cand.lab[chosen], min_area_px, min_width_px)
    labels, regions, clear = remove_small(labels, min_area_px, min_width_px,
                                        st.min_neck_mm * PX_PER_MM)
    color_error = float(color_distance(lab[::4, ::4], cand.lab[chosen][labels[::4, ::4]]).mean())

    # Number the colors that survived: tubes, then tints, then recipes, dark to light.
    used = np.unique(labels)
    order = sorted(used, key=lambda k: (cand.kind[chosen[k]], cand.lab[chosen[k], 0]))
    remap = np.zeros(len(chosen), int)
    remap[order] = np.arange(len(order))
    labels = remap[labels]

    n = regions.max()
    region_color = np.zeros(n + 1, int)
    region_color[regions.ravel()] = labels.ravel()
    ids = np.arange(1, n + 1)
    pos = np.zeros((n + 1, 2), int)
    pos[1:] = ndi.maximum_position(clear, regions, ids)
    radius = np.zeros(n + 1)
    radius[1:] = ndi.maximum(clear, regions, ids)
    blank = np.zeros(n + 1, bool)
    if st.blank_background and pal.white in chosen[order]:
        white = list(chosen[order]).index(pal.white)  # the white paint itself, not a tint
        edge = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
        blank[edge] = region_color[edge] == white
    return Plan(fmt, page, area, source_ppm, [int(chosen[k]) for k in order], labels, regions,
                region_color, pos, radius, raw_lab[..., 0].astype(np.float32), color_error, blank)
