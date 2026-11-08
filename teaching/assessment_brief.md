# Assessment brief. Mapping sealed and green surfaces in a Scottish town

## The task

Choose a Scottish town other than Stirling. Using the methods of Labs 1 and 2, map its sealed, green and water
surfaces from Sentinel-1 and Sentinel-2 data, test the map honestly and relate it to the town's greenspaces.

## What to submit

1. A report of at most 800 words, plus one map, one table and one figure.
2. Your notebook, which must run from top to bottom on a clean copy of the environment.

## What the report should contain

- **Study area and data.** The town, the bounding box, the season and the scenes you used, with a short justification.
- **Methods.** How you built the composite, the features you used and how you created training data.
- **Results.** The map, the confusion matrix and the share of each class in the town.
- **Validation.** Accuracy under a random split and under spatial blocks, either 1 km grid squares or H3 hexagons at
  resolution 8 or coarser, and an explanation of any gap between them.
- **Greenspace overlay.** A table that summarises your classes within OS Open Greenspace site types, with a short
  interpretation.
- **Limitations.** At least two limitations of the reference labels or the method, and how you would address them.

## Learning outcomes assessed

1. Find, load and pre-process Sentinel-1 and Sentinel-2 data for a study area.
2. Train and evaluate a land cover classifier and interpret a confusion matrix.
3. Combine raster and vector data in a GIS overlay.
4. Choose and justify a validation design for spatial data and explain why random splits can mislead.

## Guidance

- Small towns give small samples. If a class has too few training squares, say so and explain how it affects your
  results.
- Report the figure you would give to a planning officer, and say why.
- A clear map with a legend, a scale and a north arrow matters as much as a high accuracy.

## Generative AI

Follow the module's policy on generative AI. If you use such tools, add a short appendix that states what you used
them for. You remain responsible for every number and claim in your report.

## Weighting and deadline

Set by the module.

## Marking

See `marking_rubric.md`.
