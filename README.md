# Malen nach Zahlen

Turns your own photos into paint-by-numbers templates in the colors of your own paints.

```bash
python app.py                         # browser app on http://127.0.0.1:8765
python malen.py foto.jpg              # command line: compares A4/A3/A2, recommends one, asks which to make
python malen.py foto.jpg --format A3  # skip the comparison
```

Options: `--farben 30` (max colors), `--rezepte 8` (max recipe mixes; tints don't count),
`--min-breite 2.5` (every area at least this wide at its widest point, mm; at least 2),
`--min-engstelle 1.5` (narrower spikes and necks go to the neighbouring areas, mm),
`--palette palettes/....json`.

The browser app: drop in a photo, see A4, A3 and A2 side by side at their true relative size
with the recommendation, pick one, compare the numbered template with the painted preview,
check the color list, and export a ZIP (PDF, SVG template, preview PNG, color list CSV).
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
- `<format>_vorschau.png`: preview in the paint colors

## How it works

1. Crop the photo to the format, fit its light/dark range into what the paints can reach.
2. Choose the colors jointly from tubes, tints (paint + white 1:1, 1:3) and recipes
   (up to 2 paints + white), mixed with a Kubelka-Munk model. A tint must be about 1 ΔE
   closer than the best tube, a recipe 2.5 ΔE (`KIND_PENALTY` in `pbn/paints.py`).
3. Merge areas that are too small or thin into their most similar neighbour, smallest first.
   Then a final check (`remove_small` in `pbn/pipeline.py`) guarantees every area is
   ≥ 8 mm² and ≥ 2.5 mm wide at its widest point, and hands parts narrower than 1.5 mm
   (spikes, necks) to the neighbour they share the longest border with. Distances are
   measured to the boundary between pixels, accurate to about one pixel (0.2 mm).
4. Place each number at the widest point of its area.
5. Recommendation: the smallest format after which a bigger one keeps fewer than 5 more
   percentage points of the photo's contours, among formats the photo resolution supports
   (≥ 3 px/mm). Constants in `pbn/recommend.py`.

## Status

- `palettes/platzhalter.json` holds guessed colors. Preview, recipes and grams become
  meaningful once the swatch sheet is measured.
- Every paint has the same tinting strength until a tint sheet is measured (`S` per paint
  in the palette file).
- Paint amounts assume 0.03 g/cm² for two coats plus 30 % (`GRAMS_PER_CM2`).
