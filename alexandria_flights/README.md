# Assignment 2: Where Am I?

This folder has everything you need: four flights, and code that already runs from start to finish with its
key pieces missing. You fill in those pieces, check each one, run the part scripts, and put what they write
into your report. The handout (the PDF) explains each question; this file explains the practical side:
setup, data, scripts, checks and rules.

## 1. Setup

You need Python 3.9 or newer. Make a virtual environment for the assignment and install three packages
into it:

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
python -m pip install numpy matplotlib opencv-python
```

On macOS and Linux, type `python3` in the first line if `python` is not found; once the environment is
active, `python` works. If you already had an older OpenCV and `import cv2` fails with
`numpy.core.multiarray failed to import`, run `python -m pip install -U opencv-python`.

Then go into the code folder and run the checks. Every question says "not done yet", as it should before you
start:

```bash
cd code
python check.py
```

Run every command in this file from inside `code/`.

## 2. What is in here

```
README.md                  this file
corniche/  loop/  lost_at_sea/   synthetic flights: frames/ and gt/ (ground truth)
vpair/                     a real flight: frames/ only, no ground truth
code/
  features.py              Q1.1, Q1.2   corners, descriptors, matching          <- you fill in TODOs
  fitting.py               Q2.1, Q2.2   fitting transforms, RANSAC              <- you fill in TODOs
  odometry.py              Q3.1         chaining the flight                     <- you fill in a TODO
  mapping.py               Q3.3         the map                                 <- you fill in a TODO
  loop_closure.py          Q4.1, Q4.2   revisits, correcting the whole flight   <- you fill in TODOs
  part1.py ... part4.py    run a whole part and write its figures and tables to results/
  check.py                 sanity checks, one per question
  gt.py                    ground-truth helpers (used by check.py and the part scripts)
  viz.py                   the plots
  utils.py                 loading frames, applying transforms, small helpers
results/                   created by the part scripts
```

You edit only the five files marked above: fill in their `TODO` blocks, and add helper functions of your own
inside those files if you like. A `TODO` block looks like this, and you replace the `raise` line with your
code:

```python
    # ---------------------------------------------------------------- TODO 1.1a ----
    # 1. Ix, Iy: blur with a Gaussian of sigma sd, then differentiate ...
    raise NotImplementedError('TODO 1.1a in features.py, harris_response(): compute R')
    # ---------------------------------------------------------------- end TODO -----
```

The comments in each block say what the variables around it are and what the block must produce. Read the
function's docstring too: it gives the shape of every argument and return value.

## 3. The data

| Flight | Frames | Distance | What happens |
|---|---|---|---|
| `corniche` | 69 | 19 km | Montaza west along the coast to Mex |
| `loop` | 138 | 41 km | City, farmland and Lake Mariout, ending where it started. The aircraft climbs about 1 km |
| `lost_at_sea` | 75 | 22 km | Out over open sea and back to the coast |
| `vpair` | 200 | 9 km | A real flight over forest, suburbs, fields and the Rhine (800 x 600, no ground truth) |

Each synthetic flight has `gt/H_frame_to_map.npy`, one homography per frame from its pixels to a common
ground map at 10 m per pixel, and `gt/poses.csv` (where each frame's center lands on that map, the altitude,
the heading). `gt.py` reads them for you and gives what you need to check your results. For example, with
`LOOP = utils.flight('loop')`:

- `gt.gt_H(LOOP, i, j)` is the true transform from frame i to frame j;
- `gt.gt_path(LOOP)` is the true `T[k]` of every frame;
- `gt.path_errors(LOOP, T)` is the error of every frame of your path, in meters.

What makes the flights hard: the aircraft tilts up to about 10° sideways in turns and a little forward and
back, the altitude changes, exposure and white balance change slowly, there is vignetting, noise and JPEG
compression, a few frames are blurred, and water is different in every frame, like real water.

## 4. How to work

For every question:

1. Read the question in the handout.
2. Fill in its `TODO` blocks.
3. Run its check: `python check.py 1.1`. Each line says PASS or FAIL with what it measured. Fix what
   fails and run the check again until every line says PASS.
4. When all the questions of a part pass, run the part's script: `python part1.py`.
5. Put what the script wrote (in `results/part1/`) into your report, with the answers the question asks for.

| Question | Fill in | Check | Then run | It writes |
|---|---|---|---|---|
| 1.1 Harris Corners | `features.py`: TODO 1.1a (`harris_response`), 1.1b (`detect`) | `check.py 1.1` | `part1.py` | `results/part1/q1_1_*` |
| 1.2 Descriptors and Matching | `features.py`: TODO 1.2a, 1.2b (`describe`), 1.2c (`match`) | `check.py 1.2` | `part1.py` | `results/part1/q1_2_*` |
| 2.1 Fitting Transforms | `fitting.py`: TODO 2.1a (`fit_similarity`), 2.1b, 2.1c (`fit_homography`) | `check.py 2.1` | `part2.py` | `results/part2/q2_1_fits.md` |
| 2.2 RANSAC | `fitting.py`: TODO 2.2 (`ransac`) | `check.py 2.2` | `part2.py` | `results/part2/q2_2_ransac.md` |
| 3.1 Chaining the Flight | `odometry.py`: TODO 3.1 (`chain`) | `check.py 3.1` | `part3.py` | `results/part3/q3_1_*` |
| 3.2 Similarity or Homography? | nothing new | | `part3.py` | `results/part3/q3_2_*` |
| 3.3 The Map | `mapping.py`: TODO 3.3 (`make_map`) | `check.py 3.3` | `part3.py` | `results/part3/q3_3_*` |
| 4.1 Finding Revisits | `loop_closure.py`: TODO 4.1 (`revisits`) | `check.py 4.1` | `part4.py` | `results/part4/q4_1_*` |
| 4.2 Correcting the Whole Flight | `loop_closure.py`: TODO 4.2 (`measurement_rows`) | `check.py 4.2` | `part4.py` | `results/part4/q4_2_*` |

### What the scripts write

Everything lands in `results/`, named after its question, ready for the report:

| File | Question | What it is |
|---|---|---|
| `part1/q1_1_harris.png` | 1.1 | a corniche frame, its Harris response, and the kept corners |
| `part1/q1_2_corniche_20_21_{raw,oriented}.png` | 1.2 | matches between corniche frames 20 and 21, with the raw and with the oriented patch |
| `part1/q1_2_loop_3_132_{raw,oriented}.png` | 1.2 | the same for loop frames 3 and 132 (same streets, aircraft pointing in very different directions) |
| `part2/q2_1_fits.md` | 2.1 | the error of each fit against the ground truth, on the correct matches only |
| `part2/q2_2_ransac.md` | 2.2 | RANSAC on all the matches: error, iterations and inliers per model |
| `part3/q3_1_errors.md` | 3.1 | mean and final error of every synthetic flight |
| `part3/q3_1_path_<flight>.png` | 3.1 | the path against the truth, and the error against distance flown |
| `part3/q3_1_inliers_<flight>.png` | 3.1 | the inliers of every step |
| `part3/q3_2_models.md` | 3.2 | similarity against homography on corniche and loop |
| `part3/q3_2_paths_loop.png` | 3.2 | both paths on loop, against the truth |
| `part3/q3_2_error_vs_radius.png` | 3.2 | match errors against distance from the image center |
| `part3/q3_3_map_<flight>.jpg` | 3.3 | the map of every flight, vpair included |
| `part3/q3_3_closeup_{loop,lost_at_sea}.jpg` | 3.3 | where loop returns to its start, and where lost_at_sea finds the coast |
| `part4/q4_1_revisits_<flight>.png` | 4.1 | the accepted revisits, drawn on the chained path |
| `part4/q4_2_errors.md` | 4.2 | mean and final error before and after the correction |
| `part4/q4_2_path_<flight>.png` | 4.2 | the path before and after, against the truth |
| `part4/q4_2_map_<flight>_{before,after}.jpg` | 4.2 | the maps before and after |
| `part4/q4_2_closeup_<flight>_{before,after}.jpg` | 4.2 | where each flight returns to known ground, before and after |

Each script also prints its tables. `part3.py` and `part4.py` also save the transforms they computed as
`.npy` files.

### Tips

**Order.** Each part uses the ones before it: Part 2 needs your matching, Part 3 needs Parts 1 and 2, and
Part 4 needs Part 3. Three checks do not need the earlier parts, though: 2.1 uses synthetic points, 3.3 uses
the true path, and 4.2 uses synthetic poses. So a team can split the work from day one: one teammate on Part 1, one on
Part 2, and one on Parts 3 and 4 (3.3 and 4.2 first, then 3.1 and 4.1 once Parts 1 and 2 pass).

**If a script stops.** A part script that reaches a `TODO` you have not filled in stops and names it:

```
Stopped: TODO 3.3 in mapping.py, make_map(): warp each frame onto the canvas.
```

**`python check.py`** with no question runs every check. Its last line counts the checks that passed and
names any question that is not done yet. Run it before you submit.

**Running times** on a typical laptop, once everything is filled in: each check takes a few seconds,
`part1.py` and `part2.py` about 10 seconds, `part3.py` 3 to 4 minutes (most of it the real flight's map), and
`part4.py` about a minute.

**Looking at intermediate results.** The plot functions in `viz.py` also work on your intermediate results,
for example
`viz.show_corners(img, detect(img), harris_response(img), 'corners.png')`.

**If your Part 1 features do not work well enough**, use OpenCV's SIFT from Part 3 on, and say so in your
report. Add `--sift` to `part3.py` and `part4.py`, and to the checks that use your features (2.2, 3.1 and
4.1):

```bash
python check.py 2.2 3.1 4.1 --sift
python part3.py --sift
python part4.py --sift
```

## 5. Conventions

- **Points** are `(N, 2)` arrays of `(x, y)`: x is the column and y the row. Images are indexed `img[y, x]`.
- **Transforms** are `(3, 3)` arrays acting on homogeneous points, normalized so that `T[2, 2] = 1`.
  `utils.apply(T, p)` applies one to points.
- **"H takes frame i to frame j"** means `apply(H, points_in_frame_i)` gives where those points are in frame j.
- **The path:** `T[k]` takes frame-k pixels to frame-0 pixels, and `T[0]` is the identity. Frame 0's pixel
  coordinates are the coordinate system of the whole assignment.
- **Angles** from `np.arctan2` are in radians.

## 6. Rules

- Edit inside the `TODO` blocks, and add helpers of your own inside the five files if you like. Do not
  change the given code: the checks rely on it.
- Inside the `TODO`s, use NumPy and the OpenCV functions their comments name. Do not use OpenCV's corner
  detectors, matchers or transform estimators.
- The ground truth is for checking and scoring only. Your code in the five files must never read it.
- Online resources and AI tools are allowed, but every team member must be able to explain any part of what
  you submit.

## 7. Before you submit

- [ ] `python check.py`: every check passes (add `--sift` if you used the SIFT fallback).
- [ ] `part1.py` to `part4.py` all ran on your final code, so the figures match it.
- [ ] The report has, for every question, what its "In your report" asks for, and the answers.
- [ ] The report's first page has the names and IDs of the three team members.
- [ ] Zip the `code/` folder (no `results/`, no flights) and hand it in with the report PDF.

## Credits

- Synthetic flights: contain modified Copernicus Sentinel data 2025 (Sentinel-2 L2A, 18 August 2025),
  accessed through the Earth Search catalog on AWS Open Data.
- `vpair`: 200 query frames of the VPAIR dataset, converted to JPEG. M. Schleiss, F. Rouatbi and
  D. Cremers, "VPAIR: Aerial Visual Place Recognition and Localization in Large-scale Outdoor
  Environments", 2022. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
