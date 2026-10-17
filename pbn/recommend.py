"""Compare the paper formats for one photo and recommend one."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny

from .paints import MIX, TINT, TUBE, amounts
from .pipeline import PX_PER_MM

MIN_SOURCE_PPM = 3.0  # photo pixels per mm of painting; below this the photo is upscaled visibly
MINUTES_PER_AREA = 1.0  # rough painting-time assumption
# A bigger format must keep this many more of the photo's contours to be worth it.
# Judgment call.
CONTOUR_GAIN = 0.05


@dataclass
class Summary:
    fmt: str
    regions: int
    hours: float
    tubes: int
    tints: int
    recipes: int
    grams: float
    contours: float  # share of the photo's strong edges that survive as area borders
    source_ppm: float


def contours_kept(plan, width=1000):
    """Share of the photo's strong edges that have an area border within 2 px, both at
    the same size, so every format is judged as if viewed at the same apparent size.
    Mean color error was tried first: it is swamped by texture (grass, fur) and barely
    tells A4 from A3 even when A3 visibly keeps a face."""
    size = (width, round(width * plan.labels.shape[0] / plan.labels.shape[1]))
    L = cv2.resize(plan.photo_L, size, interpolation=cv2.INTER_AREA)
    edges = canny(L, sigma=2.0, low_threshold=4, high_threshold=10)
    border = ndi.morphological_gradient(plan.labels, size=3) > 0
    border = cv2.resize(border.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    return float((ndi.binary_dilation(border, iterations=2) & edges).sum() / max(edges.sum(), 1))


def summarize(plan, cand):
    kinds = cand.kind[plan.colors]
    area_cm2 = np.bincount(plan.labels.ravel(), minlength=len(plan.colors)) / PX_PER_MM ** 2 / 100
    grams = sum(amounts(cand.recipes[c], a)[0] for c, a in zip(plan.colors, area_cm2))
    return Summary(plan.fmt, plan.n_regions, plan.n_regions * MINUTES_PER_AREA / 60,
                   int((kinds == TUBE).sum()), int((kinds == TINT).sum()),
                   int((kinds == MIX).sum()), grams, contours_kept(plan), plan.source_ppm)


def recommend(summaries):
    """summaries: list ordered small -> large. Returns (format, reasons)."""
    reasons = []
    ok = [s for s in summaries if s.source_ppm >= MIN_SOURCE_PPM]
    for s in summaries:
        if s.source_ppm < MIN_SOURCE_PPM:
            reasons.append(f"Für {s.fmt} ist das Foto zu klein ({s.source_ppm:.1f} Pixel/mm, "
                           f"empfohlen ≥ {MIN_SOURCE_PPM:.0f}); es wird hochskaliert und wirkt weich.")
    if not ok:
        reasons.append(f"{summaries[0].fmt} wird am wenigsten hochskaliert. Besser: ein Foto "
                       f"mit höherer Auflösung.")
        return summaries[0].fmt, reasons
    pick = ok[0]
    for s in ok[1:]:
        if s.contours - pick.contours > CONTOUR_GAIN:
            reasons.append(f"{s.fmt} erhält deutlich mehr Konturen des Fotos als {pick.fmt} "
                           f"({s.contours:.0%} statt {pick.contours:.0%}).")
            pick = s
        else:
            reasons.append(f"{s.fmt} erhält kaum mehr Konturen als {pick.fmt} "
                           f"({s.contours:.0%} statt {pick.contours:.0%}), "
                           f"aber {s.hours / pick.hours:.1f}× so viel Arbeit.")
    return pick.fmt, reasons
