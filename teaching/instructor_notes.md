# Instructor notes

These notes support the two practicals in this repository. Each lab fills a two-hour computer class. They suit a GIS
or remote sensing module at Honours level (SCQF level 10) or a master's module (SCQF level 11).

## Before the class

- Build the environment from `requirements.txt` or `environment.yml` on the teaching machines, and run both notebooks
  once from the top. Lab 1 takes a few minutes, most of it downloading. Lab 2 takes about a minute without the
  optional extension.
- If the network in the room is slow, run Lab 1 in advance and give students a copy of the `data/` folder. Lab 1 then
  only needs the satellite search and loading cells to reach the Planetary Computer.
- Students do not need any accounts. The optional AlphaEarth section in Lab 2 needs a Google Earth Engine account and
  a Cloud project, so it suits a demonstration or an MSc extension.
- Ask students to bring a notebook or a sheet for their predictions. Writing the prediction down is what makes the
  Predict step work.

## Lab 1. Sealed and green surfaces around Stirling

### Suggested timing

| Minutes | Section |
| --- | --- |
| 0 to 10 | Introduction, study area and the five steps |
| 10 to 30 | Sections 1 and 2. Image search, cloud masking and the reflectance offset |
| 30 to 45 | Section 3. Spectral indices and the four places |
| 45 to 60 | Section 4. Radar |
| 60 to 75 | Sections 5 and 6. Reference labels and training squares, opened in QGIS |
| 75 to 90 | Section 7. Random forest, confusion matrix and the optical only test |
| 90 to 100 | Section 8. The map and the GeoTIFF |
| 100 to 115 | Section 9. Overlay with OS Open Greenspace |
| 115 to 120 | Make task and reflection |

### Expected results (run of 8 November 2026)

- Five Sentinel-2 scenes between June and August 2025 have less than 35% cloud. The composite is 919 rows by 1,020
  columns, and 240 pixels have no clear observation.
- NDVI at the four places. Airthrey Loch minus 0.09, the railway station 0.07, farmland on the Carse of Stirling 0.89
  and the woodland at the Wallace Monument 0.89. MNDWI is clearly positive only for the loch (0.44).
- 140 training squares with 3,340 pixels (60 sealed, 60 green and 20 water squares).
- Overall accuracy 0.977 and macro F1 0.981 on the random 30% test set. The optical bands alone also give 0.977.
- The map is about 81% green, 15% sealed and 4% water.
- Green share by greenspace type ranges from about 63% for tennis courts to 99% for golf courses.

Results can shift slightly if the Planetary Computer reprocesses scenes or if you change the season.

### Points for discussion

- **Band 8.** Vegetation is brightest in the near infrared and water darkest. Roofs and roads fall in between, which
  is why NDVI separates them from green space but not perfectly from each other.
- **The reflectance offset.** Ask what happens to NDVI if the offset of 1000 is not removed. Both bands rise, so NDVI
  is pulled towards zero, and dark surfaces suffer most.
- **The four places.** The loch is the only place with positive MNDWI. The station and the loch have similar NDVI,
  which shows why one index is not enough.
- **Radar.** Calm water is dark because it reflects the pulse away from the sensor. Buildings are bright because walls
  and the ground form corner reflectors. Under the random split the radar bands add nothing, which students can test
  again under fair validation in Lab 2.
- **WorldCover as labels.** The labels come from another model and from 2021. Students should name at least two
  consequences, such as new housing missing from the labels and the classifier learning WorldCover's own errors.
- **Greenspace overlay.** OS Open Greenspace records the purpose of a site, while the satellite records its cover.
  Tennis courts and sports facilities contain hard surfaces, and small sites cover only a few 10 m pixels.

## Lab 2. Why random splits flatter your map

### Suggested timing

| Minutes | Section |
| --- | --- |
| 0 to 10 | Recap of Lab 1 and the question of the day |
| 10 to 25 | Section 2. Moran's I for NDVI |
| 25 to 55 | Sections 3 and 4. Four validation schemes, the H3 map and the coordinates only model |
| 55 to 70 | Section 5. Block size experiment with squares and hexagons |
| 70 to 90 | Section 6. Where the errors are |
| 90 to 110 | Optional AlphaEarth extension, demonstrated or run by MSc students |
| 110 to 120 | Reflection and introduction of the assessment |

### Expected results (run of 8 November 2026)

| Model | Random pixels | By square | 1 km blocks | H3 hexagons, res. 8 |
| --- | --- | --- | --- | --- |
| Sentinel-1 and Sentinel-2 | 0.981 | 0.963 | 0.957 | 0.958 |
| Coordinates only | 1.000 | 0.631 | 0.604 | 0.653 |

- Moran's I for NDVI at the training pixels is 0.94 (pseudo p-value 0.001).
- As square blocks grow from 250 m to 4 km, the coordinates only model falls from about 0.70 to 0.55. With hexagons
  from resolution 10 to 6 (widths of about 120 m to 6 km) it falls from about 0.78 to 0.52. The Sentinel model stays
  near 0.96 throughout. The 140 training squares fall into 84 hexagons at resolution 8.
- Under 1 km blocks, 32 of the 140 squares contain at least one error. Moran's I for the square error rates is 0.05
  (pseudo p-value 0.09), so there is no clear clustering at this scale. Green squares carry the highest error rate.
- With Earth Engine, the 2025 AlphaEarth embeddings reach 0.980 under 1 km blocks, against 0.957 for the Sentinel
  bands and indices and 0.981 for both together.

### Points for discussion

- **Why the coordinates only model wins under random pixels.** Pixels from the same square share a label and almost
  the same coordinates, so a test pixel can be matched to its training neighbours by location alone. Holding out whole
  squares removes this shortcut.
- **Which number to report.** The blocked figure is the honest estimate for mapping new places. The random figure
  describes how well the model remembers its training areas.
- **Why the Sentinel model barely drops.** Sealed, green and water surfaces differ strongly in their spectra, so the
  model generalises well. The gap grows with harder classes, sparser labels or regression targets, as in the PE&RS
  study behind this lab.
- **Squares or hexagons.** In a hexagonal grid every neighbour shares an edge and lies at the same distance from the
  centre, while square blocks also touch at corners. H3 is hierarchical and covers the globe in one system, which helps
  when a study crosses national grids. Resolution 8 hexagons are about a quarter smaller than 1 km squares (0.74
  against 1 km²), which is one reason why the coordinates only model scores slightly higher under them. Students should
  compare grids of similar size before drawing conclusions about shape.
- **AlphaEarth.** Ask why the embeddings might score higher. They summarise a whole year of observations from several
  sensors, while our features come from one summer. Students should also ask whether embeddings and WorldCover share
  information, which would make the comparison easier than it looks.

## Common problems

- **Downloads fail or time out.** Run the cell again. The Planetary Computer occasionally refuses a request.
- **All NDVI values are close to zero.** The reflectance offset was not removed, or older scenes were mixed with newer
  ones.
- **Messages about `apszKnownConfigOptions` when importing rasterio.** GDAL prints these under some system locales,
  for example Turkish. Start Jupyter with `LANG=C.UTF-8` and they disappear. They do not affect the results.
- **No module named h3.** Install it with `pip install h3`. The labs need version 4 or later, because older versions
  use different function names.
- **GroupKFold fails.** Each scheme needs at least five groups. Very large blocks over a small study area can leave
  fewer groups than folds.
- **Earth Engine errors.** Check that `earthengine authenticate` has been run and that `EE_PROJECT` names a Cloud
  project registered for Earth Engine.

## Adapting the labs

- **Honours (SCQF level 10).** Run both labs as written and set the assessment brief as the coursework.
- **MSc (SCQF level 11).** Add the AlphaEarth extension, ask for a short critical review of WorldCover as reference
  data, and require a comparison of at least two block sizes in the assessment.
- **Shorter sessions.** Lab 1 sections 4 and 9 and Lab 2 sections 5 and 6 can be set as independent study.
- **Accessibility.** The class colours (grey, teal and blue) differ in lightness as well as hue. Most figures come
  with a table of the same numbers, which helps students who use screen readers.
- **Field link.** On a field course, students can photograph a few training squares with a 360 degree camera or a
  drone and compare them with the WorldCover labels.
