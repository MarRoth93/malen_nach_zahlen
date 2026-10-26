"""Pot list: a color filled into a numbered pot keeps that number in every later template."""
import itertools
import json
import os
import threading
from pathlib import Path

from .paints import describe

MAX_POTS = 99  # templates leave room for 2-digit numbers (MIN_WIDTH_FLOOR_MM)


def recipe_key(pal, recipe):
    """A color by what goes into it, in palette field numbers: ((field, parts), ...), sorted.
    Field numbers stay the same when the swatch sheet is measured again."""
    return tuple(sorted((pal.nr[i], parts) for i, parts in recipe))


def pot_candidates(keys, pal, cand):
    """Candidate indices of the colors in pots (keys from Pots.keys())."""
    wanted = set(keys)
    return [i for i, r in enumerate(cand.recipes) if recipe_key(pal, r) in wanted]


def plan_keys(plan, pal, cand):
    """Recipe keys of the painted colors, in the order of plan.painted_colors."""
    return [recipe_key(pal, cand.recipes[plan.colors[k]]) for k in plan.painted_colors]


class Pots:
    """The pot list in a JSON file, shared by the app and malen.py; re-read when it changes."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self._mtime, self._pots = None, []

    def _load(self):
        mtime = self.path.stat().st_mtime_ns if self.path.exists() else None
        if mtime != self._mtime:
            self._pots = json.loads(self.path.read_text(encoding="utf-8"))["pots"] if mtime else []
            self._mtime = mtime

    def _save(self):
        self._pots.sort(key=lambda p: p["nr"])
        lines = ",\n    ".join(json.dumps(p, ensure_ascii=False) for p in self._pots)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text('{\n  "pots": [\n    ' + lines + "\n  ]\n}\n", encoding="utf-8")
        os.replace(tmp, self.path)  # never a half-written list
        self._mtime = self.path.stat().st_mtime_ns

    @property
    def version(self):
        """Changes whenever the list changes; numbers shown in templates depend on it."""
        with self.lock:
            self._load()
            return self._mtime

    def all(self):
        with self.lock:
            self._load()
            return [dict(p) for p in self._pots]

    def keys(self, in_use_only=False):
        """Recipe keys of the pots; in_use_only: without the pots ticked off as "not to use"
        (they keep their number, but new templates no longer prefer them)."""
        with self.lock:
            self._load()
            return [tuple(map(tuple, p["recipe"])) for p in self._pots
                    if not in_use_only or p.get("in_use", True)]

    def set_in_use(self, nr, in_use):
        with self.lock:
            self._load()
            for p in self._pots:
                if p["nr"] == nr:
                    p["in_use"] = bool(in_use)
                    self._save()
                    return True
            return False

    def set_all_in_use(self, in_use):
        with self.lock:
            self._load()
            for p in self._pots:
                p["in_use"] = bool(in_use)
            if self._pots:
                self._save()

    def _in_use(self):
        """Pots that count. Unticked ones ("verwenden" off) are ignored: their numbers are free."""
        return [p for p in self._pots if p.get("in_use", True)]

    def assign(self, keys):
        """Number for each color: its pot's, or else the lowest free numbers in order.
        Also returns whether each color is already in a pot. Only pots in use count."""
        with self.lock:
            self._load()
            by_key = {tuple(map(tuple, p["recipe"])): p["nr"] for p in self._in_use()}
            used = {p["nr"] for p in self._in_use()}
            free = (n for n in itertools.count(1) if n not in used)
            numbers = [by_key[k] if k in by_key else next(free) for k in keys]
            return numbers, [k in by_key for k in keys]

    def assign_plan(self, plan, pal, cand):
        """(number, in pot) per entry of plan.colors; colors that are not painted get (None, False)."""
        numbers, in_pot = [None] * len(plan.colors), [False] * len(plan.colors)
        for k, n, have in zip(plan.painted_colors, *self.assign(plan_keys(plan, pal, cand))):
            numbers[k], in_pot[k] = n, have
        return numbers, in_pot

    def register_plan(self, plan, pal, cand, numbers, source):
        ks = plan.painted_colors
        srgb = ["#" + "".join(f"{round(v):02x}" for v in cand.srgb[plan.colors[k]]) for k in ks]
        texts = [describe(pal, cand.kind[plan.colors[k]], cand.recipes[plan.colors[k]]) for k in ks]
        self.register(plan_keys(plan, pal, cand), [numbers[k] for k in ks], srgb, texts, source)

    def register(self, keys, numbers, srgb_hex, texts, source):
        """Remember the colors of an exported template that are not in a pot (in use) yet.
        An unticked pot with the same number or recipe is replaced by the new one."""
        with self.lock:
            self._load()
            known = {tuple(map(tuple, p["recipe"])) for p in self._in_use()}
            used = {p["nr"] for p in self._in_use()}
            new = [(k, n, c, t) for k, n, c, t in zip(keys, numbers, srgb_hex, texts) if k not in known]
            for _, n, _, _ in new:
                if n > MAX_POTS:
                    raise ValueError(f"Keine Topfnummer mehr frei (höchstens {MAX_POTS}).")
                if n in used:  # the list changed since the numbers were handed out
                    raise ValueError(f"Topfnummer {n} ist inzwischen belegt. Bitte erneut exportieren.")
            replaced = {n for _, n, _, _ in new}
            new_keys = {k for k, _, _, _ in new}
            self._pots = [p for p in self._pots if p.get("in_use", True)
                          or (p["nr"] not in replaced and tuple(map(tuple, p["recipe"])) not in new_keys)]
            for k, n, c, t in new:
                self._pots.append({"nr": n, "recipe": [list(x) for x in k], "srgb": c,
                                   "text": t, "source": source})
            if new:
                self._save()

    def free(self, nr):
        """An emptied pot: its number is given to the next new color."""
        with self.lock:
            self._load()
            left = [p for p in self._pots if p["nr"] != nr]
            if len(left) == len(self._pots):
                return False
            self._pots = left
            self._save()
            return True


def room_error(numbers, in_pot):
    """Message when new colors would need pot numbers above MAX_POTS, else None."""
    over = sum(1 for n, have in zip(numbers, in_pot) if n is not None and not have and n > MAX_POTS)
    if over:
        return (f"Alle {MAX_POTS} Topfnummern sind belegt; für {over} neue "
                f"{'Farbe' if over == 1 else 'Farben'} fehlt Platz. Gib unter 'Töpfe' leere Töpfe frei.")
    return None
