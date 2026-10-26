# Malen nach Zahlen

Turns your own photos into paint-by-numbers templates in the colors of your own paints.

```bash
python app.py                         # browser app on http://127.0.0.1:8765
python malen.py foto.jpg              # command line: compares A4/A3/A2, recommends one, asks which to make
python malen.py foto.jpg --format A3  # skip the comparison
```

Options: `--farben 30` (max colors), `--rezepte 8` (max recipe mixes; tints don't count),
`--min-breite 2.0` (every area at least this wide at its widest point, mm; at least 2),
`--min-engstelle 1.0` (narrower spikes and necks go to the neighbouring areas, mm),
`--weiss-lassen` (white areas touching the picture's edge stay unpainted: no number, no paint;
in the app: "Weißen Hintergrund frei lassen"), `--palette palettes/....json`.

The browser app: drop in a photo, see A4, A3 and A2 side by side at their true relative size
with the recommendation, pick one, compare the numbered template with the painted preview,
check the color list, and export a ZIP (PDF, SVG template, preview PNG, color list CSV) into
`exported/` as `<photo>_<format>_<colors>-Farben_<recipes>-Rezepte.zip`; a second export of
the same gets `_2`, `_3`, … instead of overwriting. Under "Varianten vergleichen" the chosen
format can be computed with up to two other color/recipe limits and compared side by side,
with tubes, tints, recipes, areas, time and the color error (mean ΔE of the painted colors to
the smoothed photo, `Plan.color_error`); the variant clicked is the one shown and exported.
Settings (colors, recipes, minimum sizes) are under "Einstellungen". It only listens on
localhost, and photos stay in memory until the app stops ("Beenden" in the page).
Starting it again while it runs just opens the browser tab. On this laptop it is also in the
GNOME app list as "Malen nach Zahlen" (`~/.local/share/applications/malen-nach-zahlen.desktop`,
icon `gui/icon.svg`).

Command-line output in `ausgabe/<photo name>/`:

- `formatvergleich.png`: the three formats side by side with areas, time, colors and how
  many of the photo's contours survive
- `<format>_vorlage.pdf`: page 1 the outlines at 1:1 (light gray, numbered), page 2 the
  color list with recipes in grams, page 3 a preview to print as reference
- `<format>_nur_vorlage.pdf`: only the outline page, to print on its own
- `<format>_vorschau.png`: preview in the paint colors

## How it works

1. Crop the photo to the format, fit its light/dark range into what the paints can reach.
2. Choose the colors jointly from tubes, tints (paint + white 1:1, 1:3) and recipes
   (up to 2 paints + white), mixed with a Kubelka-Munk model. A tint must be about 1 ΔE
   closer than the best tube, a recipe 2.5 ΔE (`KIND_PENALTY` in `pbn/paints.py`).
3. Merge areas that are too small or thin into their most similar neighbour, smallest first.
   Then a final check (`remove_small` in `pbn/pipeline.py`) guarantees every area is
   ≥ 8 mm² and ≥ 2.0 mm wide at its widest point, and hands parts narrower than 1.0 mm
   (spikes, necks) to the neighbour they share the longest border with. Distances are
   measured to the boundary between pixels, accurate to about one pixel (0.2 mm).
4. Place each number at the widest point of its area.
5. Recommendation: the smallest format after which a bigger one keeps fewer than 5 more
   percentage points of the photo's contours, among formats the photo resolution supports
   (≥ 3 px/mm). Constants in `pbn/recommend.py`.

## Pots

Every color you mix (or fill from a tube) goes into a numbered pot, and keeps that number in
all later templates. `palettes/toepfe.json` lists the pots by recipe in palette field numbers;
it started with `DSC03020_A3_30-Farben_8-Rezepte`. A template's colors that are already in a
pot show their pot number and "schon im Topf"; new colors get the lowest free numbers and a ★
in the color list, and an export (app or `malen.py`) makes them pots. Choosing the colors
prefers pots: a color already in a pot carries no penalty, a new one 2 ΔE extra
(`NEW_POT_PENALTY`), so a pot wins when it is nearly as close. Replayed on four exports this
needed 43 % fewer new pots, with the color error up to 0.7 higher. At most 99 pots, so
numbers stay 2-digit. Under "Töpfe" in the app: untick "verwenden" (the pot keeps its
number but new templates no longer prefer it) or "löschen" (the number becomes free). Export
with "Neue Farben nicht als Töpfe speichern" (`malen.py --ohne-toepfe`) leaves the list as it is.

## Measuring the paints

```bash
python kalibrieren.py ausstriche/ --ohne 37 39 42 47   # RAW photos of the swatch sheet
```

Print `Farbkalibrierung_Ausstreichblatt.pdf` at 100 %, paint each field, write the tube code
next to it, photograph it as RAW with the gray card flat on its field. The script finds the
markers, corrects the light falloff across the sheet with the bare paper, calibrates on the
gray card (18 %, `GRAY_CARD`), and writes `palettes/gemessen.json`; names and codes already in
that file are kept. Gloss on the paint only adds light, so with several photos from different
angles each field uses its darker readings (`SHEEN_TOL`). Metallic and pearl paints change with
the angle; leave them out with `--ohne`.

## Status

- `palettes/gemessen.json` (the default) is measured from the swatch sheet in `ausstriche/`,
  without the metallic/pearl fields 37, 39, 42 and 47. `palettes/platzhalter.json` holds the
  old guesses.
- Every paint has the same tinting strength until a tint sheet is measured (`S` per paint
  in the palette file).
- Paint amounts assume 0.03 g/cm² for two coats plus 30 % (`GRAMS_PER_CM2`). A new mixed
  color is weighed out for one full pot, 10 g (`POT_G`), or more if the painting needs more;
  "Töpfe" shows the 10 g amounts of every mixed pot, for mixing or refilling it.
