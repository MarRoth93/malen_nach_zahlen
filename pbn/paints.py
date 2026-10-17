"""Paint palette, Kubelka-Munk mixing, and the candidate colors: tube, tint, recipe."""
import json
from dataclasses import dataclass
from itertools import combinations
from math import gcd

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

TUBE, TINT, MIX = 0, 1, 2
TINT_RATIOS = [(1, 1), (1, 3)]  # paint : white, by weight
MIX_WHITE_ONLY = [2, 5, 8, 12, 20]  # 1 part paint + n parts white, beyond the fixed tints
MIX_PARTS = range(1, 5)  # parts of each of the two paints in a recipe
MIX_WHITE_PARTS = [0, 1, 2, 3, 5, 8, 12]

# Judgment calls, in ΔE per pixel: where a tube is within this much of a tint
# (or recipe), the tube wins. Encodes "tube first, then tint, then recipe".
KIND_PENALTY = {TUBE: 0.0, TINT: 1.0, MIX: 2.5}
# CIEDE2000 lightness factor < 1: light/dark errors count more than hue errors.
KL = 0.65

# Paint use: two coats of acrylic, plus a safety margin. A guess; weigh the pots
# before and after a first painting to calibrate it.
GRAMS_PER_CM2 = 0.03
AMOUNT_MARGIN = 1.3
MIN_PART_G = 0.3  # smallest weighable part on a 0.01 g scale (about 3 % error)
MIN_BATCH_G = 3.0


def srgb_to_linear(rgb):
    c = np.asarray(rgb, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    c = np.clip(lin, 0, 1)
    return 255 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb):
    return rgb2lab(np.asarray(rgb, float) / 255)


def color_distance(lab1, lab2):
    return deltaE_ciede2000(lab1, lab2, kL=KL)


@dataclass
class Palette:
    source: str
    nr: list[int]
    names: list[str]
    K: np.ndarray  # (P, 3) Kubelka-Munk absorption per RGB channel
    S: np.ndarray  # (P, 3) scattering per RGB channel
    white: int  # index of the white used for tints and recipes

    @property
    def is_placeholder(self):
        return self.source != "calibrated"

    def label(self, i):
        return f"{self.nr[i]} {self.names[i]}"


def load_palette(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    paints = data["paints"]
    R = np.clip(srgb_to_linear([p["srgb"] for p in paints]), 1e-3, 1 - 1e-3)
    # Without a measured tint sheet every paint gets the same scattering
    # (single-constant Kubelka-Munk), i.e. the same tinting strength.
    S = np.array([p.get("S", [1.0, 1.0, 1.0]) for p in paints], float)
    K = S * (1 - R) ** 2 / (2 * R)
    nr = [p["nr"] for p in paints]
    return Palette(data.get("source", "placeholder"), nr, [p["name"] for p in paints],
                   K, S, nr.index(data["white_nr"]))


def mix_reflectance(pal, idx, weights):
    """Linear-RGB reflectance of mixes. idx, weights: (M, k); weights sum to 1 per row."""
    K = np.einsum("mk,mkc->mc", weights, pal.K[idx])
    S = np.einsum("mk,mkc->mc", weights, pal.S[idx])
    q = K / S
    return 1 + q - np.sqrt(q * q + 2 * q)


@dataclass
class Candidates:
    kind: np.ndarray  # (N,) TUBE / TINT / MIX
    recipes: list  # per candidate: ((paint index, parts), ...)
    srgb: np.ndarray  # (N, 3) 0-255
    lab: np.ndarray  # (N, 3)


def build_candidates(pal):
    others = [i for i in range(len(pal.nr)) if i != pal.white]
    w = pal.white
    recipes, kinds = [((i, 1),) for i in range(len(pal.nr))], [TUBE] * len(pal.nr)
    for i in others:
        for a, b in TINT_RATIOS:
            recipes.append(((i, a), (w, b)))
            kinds.append(TINT)
        for b in MIX_WHITE_ONLY:
            recipes.append(((i, 1), (w, b)))
            kinds.append(MIX)
    for i, j in combinations(others, 2):
        for a in MIX_PARTS:
            for b in MIX_PARTS:
                for c in MIX_WHITE_PARTS:
                    if gcd(gcd(a, b), c) > 1:  # same ratio as a smaller one
                        continue
                    recipes.append(((i, a), (j, b), (w, c)) if c else ((i, a), (j, b)))
                    kinds.append(MIX)

    idx = np.zeros((len(recipes), 3), int)
    wt = np.zeros((len(recipes), 3))
    for r, rec in enumerate(recipes):
        total = sum(p for _, p in rec)
        for k, (i, p) in enumerate(rec):
            idx[r, k], wt[r, k] = i, p / total
    srgb = linear_to_srgb(mix_reflectance(pal, idx, wt))
    return Candidates(np.array(kinds), recipes, srgb, srgb_to_lab(srgb))


def describe(pal, kind, recipe):
    if kind == TUBE:
        return f"Tube {pal.label(recipe[0][0])}"
    if kind == TINT:
        (i, a), (_, b) = recipe
        return f"{pal.label(i)} + Weiß {a}:{b}"
    return " + ".join(f"{p} T. {pal.label(i) if i != pal.white else 'Weiß'}"
                      for i, p in recipe)


def amounts(recipe, area_cm2):
    """Grams needed for an area, and the grams of each part of the batch to weigh out."""
    need = area_cm2 * GRAMS_PER_CM2 * AMOUNT_MARGIN
    if len(recipe) == 1:
        return need, []
    frac = np.array([p for _, p in recipe], float)
    frac /= frac.sum()
    batch = max(need, MIN_BATCH_G, MIN_PART_G / frac.min())
    return need, list(frac * batch)
