# Rainfall Downscaling

Learning a generative map from coarse ERA5 reanalysis (0.25°, ~25 km) to CHIRPS-resolution
daily precipitation (0.05°, ~5 km) over Uttarakhand, India. Three approaches are compared: a
tile-wise linear regression baseline, a CycleGAN, and a U-Net conditional GAN.

Coursework project for EE708.

---

## Contents

- [The problem](#the-problem)
- [Data](#data)
- [Pipeline](#pipeline)
- [EDA and filtering](#eda-and-filtering)
- [Method 1: tile-wise linear regression](#method-1-tile-wise-linear-regression)
- [Method 2: CycleGAN](#method-2-cyclegan)
- [Method 3: U-Net conditional GAN](#method-3-u-net-conditional-gan)
- [Results](#results)
- [Extreme-value bias analysis](#extreme-value-bias-analysis)
- [Repository layout](#repository-layout)
- [Setup and reproduction](#setup-and-reproduction)
- [Known issues](#known-issues)
- [References](#references)

---

## The problem

Rainfall data feeds crop-yield modelling, reservoir and hydropower operation across the
Ganges, Indus and Brahmaputra basins, and flood and landslide warning in the Himalaya. Global
reanalyses like ERA5 cover the whole region but at 0.25°, where one cell spans about 25 km. In
mountain terrain that averages away most of the variability you actually care about. CHIRPS
blends satellite retrievals with station data at 0.05° and is the finer target here.

The physically rigorous route is to nest a 5 km regional model inside the global one, which
costs petabytes and months of cluster time for a multi-decade daily record. This project takes
the statistical route instead and learns the coarse-to-fine mapping directly from paired data.

The two products differ in more than sharpness. They also disagree about where the rain is:

![CHIRPS and ERA5 misalignment](images/chirps_era5_misalignment.png)

And they disagree systematically in magnitude. ERA5 overestimates light rain and
underestimates heavy rain relative to CHIRPS, which shows up as the S-shaped departure from
the 1:1 line in the baseline Q-Q plot:

![ERA5 vs CHIRPS bias](images/era5_bias.png)

## Data

Both products were pulled from Google Earth Engine as daily GeoTIFFs over a fixed Uttarakhand
bounding box.

| | CHIRPS | ERA5 |
|---|---|---|
| GEE collection | `UCSB-CHG/CHIRPS/DAILY` | `ECMWF/ERA5/DAILY` |
| Native grid over the ROI | 64 × 64 | 13 × 13 |
| Resolution | 0.05° (~5 km) | 0.25° (~25 km) |
| Bands used | precipitation | air temperature, dewpoint temperature, total precipitation, u-wind, v-wind |
| Units | mm/day | precipitation in m/day, converted ×1000 to mm/day |
| Daily files exported | 16,314 | 14,435 |
| Coverage | 1981-01-01 to ~2025-08 | 1981-01-01 to 2020-07-09 |
| Value range observed | 0 to 710.4 mm/day | |

The region of interest is the polygon `(77.8, 31.9) (81.0, 31.9) (81.0, 28.75) (77.8, 28.75)`,
3.2° × 3.15°, geodesic area 107,458 km², centred at (30.32 N, 78.88 E). ERA5's record in the
GEE `ECMWF/ERA5/DAILY` collection ends 2020-07-09, which is what caps the paired data.

ERA5 channels are stacked in this order everywhere in the repo (`utils/data_loader.py:65`):

| Index | Variable | Source band |
|---|---|---|
| 0 | air temperature | 1 |
| 1 | dewpoint temperature | 4 |
| 2 | total precipitation (×1000 → mm/day) | 5 |
| 3 | u-wind | 8 |
| 4 | v-wind | 9 |

Channel 2 is the precipitation channel every model notebook reaches for.

ERA5 is resampled from its 13 × 13 native grid onto the 64 × 64 CHIRPS grid with
`rasterio.warp.reproject(..., Resampling.cubic)` (`utils/data_loader.py:32`), preserving each
collection's own CRS and transform from export. NaNs are filled with 0 on load.

## Pipeline

```mermaid
flowchart TD
    A["Google Earth Engine<br/>CHIRPS + ERA5 daily"] -->|"01, 02"| B["GeoTIFF on disk<br/>16,314 + 14,435 files"]
    B -->|"utils/data_loader.py"| C["RainfallDataLoader<br/>NaN fill, cubic reprojection<br/>ERA5 13x13 → 64x64"]
    C -->|"03"| D["EDA<br/>seasonality, trend, ERA5 vs CHIRPS bias"]
    C -->|"04"| E["Threshold 1 mm/day<br/>cluster filter, year >= 1999<br/>519 ERA5→CHIRPS pairs"]
    C -->|"05"| F["Degrade CHIRPS to ERA5 grid<br/>and interpolate back<br/>682 CHIRPS→CHIRPS pairs"]
    E --> G["data/final_era5.pkl<br/>data/final_chirps.pkl"]
    F --> H["data/filtered_chirps_reprojected.pkl<br/>data/filtered_chirps.pkl"]
    H -->|"06"| I["Tile-wise linear regression"]
    H -->|"07 + cyclegan/"| J["CycleGAN"]
    H -->|"08"| K["U-Net conditional GAN"]
    G -->|"09"| L["GPD extreme-value bias analysis"]
    I --> M["Q-Q bias, RMSE, R²"]
    J --> M
    K --> M
```

Two paired datasets fall out of this, and the difference between them matters a lot when
reading the results further down.

The first is ERA5 to CHIRPS, 519 pairs. This is the real task: it crosses products, so the
model has to absorb ERA5's systematic bias on top of adding detail.

The second is degraded CHIRPS to CHIRPS, 682 pairs, where the input is CHIRPS pushed down to
the ERA5 grid and interpolated back up. That makes it pure super-resolution with no
cross-product bias, so it is an easier, idealised version of the problem. Every headline model
below is trained on this second set.

## EDA and filtering

Roughly 70 % of days in the record carry no meaningful rain over the ROI, and training on them
just teaches a model to output zeros. Two filters run before any model sees the data.

The cluster-area filter thresholds the field at 16, runs
`cv2.connectedComponentsWithStats(connectivity=8)`, and keeps a day only if its large clusters
cover at least 20 % of the frame. Days before 1999 are dropped outright: earlier CHIRPS over
this region is visibly blockier, does not match the later spatial statistics, and pushed
training loss up noticeably.

![Data filtering method](images/data_filtering_method.png)

![Filtering result](images/filtering_result.png)

What survives lands almost entirely in July, August and September, which is the monsoon. The
filter recovers the seasonality without being told about it:

![Filtered rainfall months](images/filtered_rainfall_months.png)

Seasonality and trend come from an HP filter on the monthly means, plus Holt exponential
smoothing over the daily spatial-mean series:

![Seasonality and trend](images/eda_seasonality_trend.png)

Daily rainfall is heavily right-skewed, and a log transform makes the distribution tractable:

![Log transform](images/log_transform_skew.png)

Method development used CHIRPS reprojected onto the 13 × 13 ERA5 grid and interpolated back to
64 × 64 as a stand-in for ERA5. Same resolution loss, none of the cross-product bias:

![Upscaled CHIRPS vs CHIRPS](images/upscaled_chirps_vs_chirps.png)

Final counts: 682 degraded-CHIRPS to CHIRPS pairs, split 545 train and 137 test
chronologically with no shuffle, and 519 ERA5 to CHIRPS pairs, of which 283 are normal-rain
days and 236 exceed the 99th-percentile threshold of 111.38 mm/day.

## Method 1: tile-wise linear regression

[`06_linear_regression.ipynb`](06_linear_regression.ipynb)

Rather than one global 4096 to 4096 regression, cut each 64 × 64 field into 64 non-overlapping
8 × 8 tiles, flatten each to a 64-vector, and fit an independent `LinearRegression` per tile
*position*. Each model learns the coarse-to-fine mapping for its own patch of terrain, and the
predictions get reshaped and reassembled into the full field.

![Linear regression architecture](images/linear_regression_architecture.png)

Post-processing clamps negative predictions to zero, since negative precipitation has no
meaning and unconstrained least squares produces it freely.

![Linear regression result](images/linear_regression_result.png)

On 137 held-out days: MSE 119.609, R² 0.705, mean bias −0.032 mm/day, RMSE 10.937 mm/day.
Mean bias is essentially zero, which is what least squares guarantees, so the Q-Q plot is the
more informative view. It tracks the 1:1 line at low intensities and falls below it in the
tail, meaning the method underestimates heavy rainfall.

![Linear regression bias](images/linear_regression_bias.png)

An earlier variant in [`archive/regression_downscaling.ipynb`](archive/regression_downscaling.ipynb)
fit 256 tile models directly on ERA5 to CHIRPS over 10,592 unfiltered days and reached only
MSE 55.066, R² 0.104. The lower MSE there is an artifact of the unfiltered set being mostly
dry days; the R² is the number that matters, and it says the cross-product mapping is a much
harder problem.

## Method 2: CycleGAN

[`07_cyclegan_stabilized.ipynb`](07_cyclegan_stabilized.ipynb) · [`cyclegan/`](cyclegan/)

CycleGAN learns two generators, coarse to fine and fine to coarse, plus two discriminators,
tied together by a cycle-consistency loss. It needs no paired samples.

![CycleGAN architecture](images/cyclegan_architecture.png)

Two tracks were run. The first went through the reference implementation:
[`cyclegan/`](cyclegan/) holds the custom dataset class, the driver notebook, and a patch
against [`junyanz/pytorch-CycleGAN-and-pix2pix`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
at `2a7afba`. ResNet-9-block generators at 11.366 M params each, 4-layer PatchGAN
discriminators at 6.958 M each, instance norm, LSGAN loss, `--preprocess none`. The exact
commands and their caveats are in [`cyclegan/README.md`](cyclegan/README.md).

The second is a self-contained reimplementation in `07_cyclegan_stabilized.ipynb` that runs
without the upstream repo: `ResnetGenerator(ngf=64, n_blocks=6)`, `NLayerDiscriminator` with
5 layers, `ImagePool(50)`, `GANLoss("lsgan")`, `EPOCHS=15`, `BATCH_SIZE=16`,
`LAMBDA_A=LAMBDA_B=10`, `IDT_MULT=0.5`, `LR_G=2e-4`, `LR_D=2e-5`, linear decay from epoch 7,
seed 88. Final epoch: `D_A 0.3051, D_B 0.2640, G 2.1203`.

![CycleGAN loss](images/cyclegan_loss.png)

![CycleGAN result](images/cyclegan_result.png)

The losses converge cleanly but the bias does not improve with them. The unpaired objective
has no term forcing pixel-to-pixel agreement with the target, only a requirement that outputs
*look like* CHIRPS, and the Q-Q plot departs from the 1:1 line across the whole range:

![CycleGAN bias](images/cyclegan_bias.png)

That is roughly what you would expect. Downscaling is a paired, pixel-aligned problem, whereas
CycleGAN is built for unpaired domain transfer, where altering structure is a feature rather
than a bug.

| | U-Net conditional GAN | CycleGAN |
|---|---|---|
| Translation | paired | unpaired |
| Input to output coupling | direct, pixel-to-pixel | loose, via cycle consistency |
| Generator | encoder-decoder with skip connections | encoder, ResNet, decoder, no skips |
| High-frequency detail | carried across by skips | not directly |
| Built for | low to high resolution, denoising, downscaling | style and domain transfer |
| Structure preservation | strong | may alter structure by design |

## Method 3: U-Net conditional GAN

[`08_unet_gan.ipynb`](08_unet_gan.ipynb) holds the best-performing model here.

It is a pix2pix-style conditional GAN with a 4-level U-Net generator. The skip connections
carry high-frequency spatial detail straight from encoder to decoder, which matters because a
downscaling model cannot afford to lose that detail at the bottleneck.

![U-Net GAN architecture](images/unet_gan_architecture.png)

The generator has four stride-2 `Conv2d(k=4, s=2, p=1)` down blocks, 1 → 64 → 128 → 256 → 512,
each with `InstanceNorm2d` and `LeakyReLU(0.2)`, then four `ConvTranspose2d` up blocks with
the corresponding encoder activation concatenated in, and a `Tanh` output. The discriminator
is a 5-conv PatchGAN over `cat([input, target], dim=1)`, emitting a real/fake score per patch
rather than one per image.

Training uses `BCEWithLogitsLoss` for the adversarial term plus `100 × L1Loss` for
reconstruction, Adam with `betas=(0.5, 0.999)`, `EPOCHS=15`, `BATCH_SIZE=16`, `LR_G=2e-4`,
`LR_D=2e-5`, seed 88, 43 batches per epoch.

| Epoch | D loss | G loss |
|---|---|---|
| 1 | 21.655 | 1465.620 |
| 5 | 0.790 | 590.947 |
| 10 | 0.249 | 500.248 |
| 15 | 0.121 | 513.717 |

The left panel below is an earlier, unstable configuration with equal learning rates and no L1
weighting; the right panel is the run above. Dropping the discriminator learning rate an order
of magnitude below the generator's is what stopped the oscillation.

![U-Net GAN loss](images/unet_gan_loss.png)

![U-Net GAN result](images/unetgan_result.png)

![U-Net GAN bias](images/unetgan_bias.png)

## Results

| Model | Input to target | Days | Metrics |
|---|---|---|---|
| ERA5, no model (baseline) | ERA5 vs CHIRPS daily means | 14,435 | mean bias 0.836, RMSE 5.639 |
| Linear regression, 256 tiles | ERA5 to CHIRPS | 10,592 | MSE 55.066, R² 0.104 |
| Linear regression, 64 tiles | degraded CHIRPS to CHIRPS | 682 | MSE 119.609, R² 0.705, mean bias −0.032, RMSE 10.937 |
| CycleGAN | degraded CHIRPS to CHIRPS | 682 | mean bias 56.353, RMSE 6.667 † |
| U-Net conditional GAN | degraded CHIRPS to CHIRPS | 682 | mean bias 77.103, RMSE 5.448 † |

† Those two rows are not in mm/day. Both GAN pipelines evaluate through
`plot_bias_using_QQ_plot` applied to 8-bit 0 to 255 greyscale renderings of the tensors, since
the CycleGAN path round-trips through PNG on disk and the U-Net path goes through `tensor2im`.
The linear regression row is in mm/day. So the GAN and regression rows are not directly
comparable, and a mean bias larger than the RMSE, which is arithmetically impossible for a
consistent estimator, is the visible symptom of that quantisation. Within the GAN rows the
comparison does hold, and the U-Net GAN wins on both.

Reading the Q-Q plots side by side is the comparison to trust, and the ordering there is
unambiguous: raw ERA5 is worst, linear regression corrects the mean but collapses the tail,
CycleGAN drifts across the whole range, and the U-Net GAN tracks the 1:1 line furthest.

One caveat worth stating plainly. Every headline model is trained on degraded CHIRPS to
CHIRPS, not on ERA5. That removes the cross-product bias and turns the job into pure
super-resolution, which is a large part of why the losses come out as low as they do. Getting
the trained architecture to transfer to real ERA5 input is the next step, not a result claimed
here.

## Extreme-value bias analysis

[`09_bias_measurement.ipynb`](09_bias_measurement.ipynb)

Aggregate RMSE says very little about flood risk, which is concentrated in the tail of the
distribution. This notebook fits extreme-value statistics per grid cell:

1. Peaks-over-threshold at the 95th percentile, computed independently for each of the
   64 × 64 cells.
2. A Generalized Pareto Distribution fit per cell (`scipy.stats.genpareto`, `floc=0`, requiring
   at least 30 exceedances).
3. Return-level curves `RL(T) = u + σ/ξ · ((T·λ)^ξ − 1)` at roughly 18 exceedances per year.

Four sites are extracted by lon/lat to grid index: Dehradun (33, 5), Uttarkashi (24, 13),
Nainital (51, 33), Pithoragarh (47, 48). Outputs are a return-period curve per site, a 100-year
return-level bias map (±50 mm/day), a mean daily bias map (±5 mm/day), and an extreme-tail PDF
overlay above the 95th percentile.

## Repository layout

```
.
├── 01_export_chirps.ipynb          CHIRPS daily export from Earth Engine
├── 02_export_era5.ipynb            ERA5 daily export from Earth Engine
├── 03_eda.ipynb                    seasonality, trend, ERA5 vs CHIRPS bias, extremes
├── 04_data_preparation.ipynb       ERA5 to CHIRPS pairs (519) + synthetic extremes
├── 05_downscale_chirps.ipynb       degraded CHIRPS to CHIRPS pairs (682)
├── 06_linear_regression.ipynb      Method 1
├── 07_cyclegan_stabilized.ipynb    Method 2, self-contained reimplementation
├── 08_unet_gan.ipynb               Method 3, best model
├── 09_bias_measurement.ipynb       GPD extreme-value bias analysis
│
├── utils/
│   ├── constants.py                GeoTIFF directory paths, edit these first
│   ├── data_loader.py              RainfallDataLoader, reprojection, NaN fill
│   └── vizualizations.py           plot_bias_using_QQ_plot, the shared metric
│
├── cyclegan/                       custom files for the reference CycleGAN repo
├── images/                         figures used in this README
├── archive/                        exploratory and superseded work, see below
├── pyproject.toml                  dependencies, source of truth
├── uv.lock                         fully pinned resolution
├── requirements.txt                generated from uv.lock, for pip users
└── data/                           gitignored, regenerated by 04 and 05
```

`archive/` keeps the paths not taken, for provenance. Everything in it expects to be run from
the repository root:

| File | What it was |
|---|---|
| `export_CHIRPS.ipynb`, `export_ERA5.ipynb` | whole-Himalaya exports, before the ROI narrowed to Uttarakhand |
| `himalayas_precipitation.{ipynb,py}`, `himalayas_map.html` | initial region scoping with geemap |
| `reprojection.ipynb`, `plotting.ipynb` | single-file reprojection proof of concept; basemap overlays |
| `regression_downscaling.ipynb` | first-generation 256-tile regression on ERA5 to CHIRPS |
| `GAN_downscaling.ipynb`, `New_GAN_downscaling.ipynb` | early GAN attempts, 512×512 over all-India |
| `cycle_gan.ipynb`, `cycle_gan_stabilized_experiment.ipynb`, `cycle-gan-stabilized.py` | CycleGAN iterations, including a stacked two-stage variant |
| `orographicRainLinearTheory.py`, `oro_rain_*.py`, `DEM.tif` | Smith and Barstad linear orographic precipitation theory, a physics-based side track that never got wired into the ML pipeline |
| `emissions.csv` | codecarbon log: 0.0043 kWh, 0.0013 kg CO₂eq for one training session |

## Setup and reproduction

The environment is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` is the
source of truth and `uv.lock` pins the full resolved graph.

```bash
uv sync                       # creates .venv, installs everything from uv.lock
uv run jupyter lab            # launch the notebooks
```

`uv sync` will fetch CPython 3.12 itself if it is not already present, so there is nothing
else to install first. To run a single command without activating anything, prefix it with
`uv run`. To activate the shell the usual way, `source .venv/bin/activate`.

Adding or changing a dependency:

```bash
uv add <package>              # updates pyproject.toml and uv.lock together
uv lock --upgrade             # re-resolve everything to latest compatible
uv export --no-hashes --no-annotate --no-header -o requirements.txt
```

`requirements.txt` is generated from the lock for environments without uv (Colab, plain pip)
and should not be hand-edited.

The project is pinned to Python 3.12, because it was developed on 3.12.3 and the geospatial
wheels (`rasterio`, `geopandas`, `pyproj`) lag behind the newest interpreters. Training ran on
an RTX 4070 Laptop GPU. `torch` comes from the default PyPI wheels, which bundle CUDA on
Linux; for a CPU-only or different CUDA build see
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

One caveat on the lock. It resolves to current releases, which are newer than the stack the
committed notebook outputs were produced on in late 2025, in particular `pandas` 3.x,
`opencv` 5.x and `numpy` 2.5. Expect the odd deprecation to need patching, or pin those three
down if you want a bit-for-bit rerun.

**Earth Engine access.** Notebooks `01` and `02` call `ee.Initialize(project=...)` with a
Cloud project that has the Earth Engine API enabled. Run `earthengine authenticate` once, then
set your own project ID. Exports land in Google Drive as daily GeoTIFFs.

**Point the loader at your GeoTIFFs.** `utils/constants.py` expects them two levels above the
repository:

```python
CHIRPS_GEOTIFF_DIR = '../../CHIRPS_daily_Uttarakhand'
ERA5_GEOTIFF_DIR   = '../../ERA5_daily_Uttarakhand'
```

Edit these to match wherever you downloaded the exports.

**Run in order.** `03`, `04` and `05` build `data/*.pkl`; `06`, `07` and `08` train; `09`
analyses extremes. `data/` is gitignored, about 378 MB of pickles, and is fully regenerated by
`04` and `05`.

**CycleGAN** needs the upstream repository; see [`cyclegan/README.md`](cyclegan/README.md).

## Known issues

These are documented rather than quietly fixed, because the committed notebook outputs were
produced with them in place.

A filename typo splits the pipeline in two. `05_downscale_chirps.ipynb` writes
`data/filterd_chirps_reprojected.pkl`, missing the `e` in "filtered". Notebooks `07` and `08`
read that typo'd name, while `06` reads the correct `filtered_chirps_reprojected.pkl`. Only
one of the two files exists at any time, so one branch always breaks: straight after running
`05` the regression notebook fails, and after manually renaming the file the two GAN notebooks
fail instead. Fixing it takes one rename plus a one-character edit in `05`, `07` and `08`.

Cubic resampling overshoots at sharp edges and produces negative precipitation, down to
−19.03 mm/day in the reprojected ERA5. Downstream code either thresholds at 1 mm/day or clamps
at 0, but nothing corrects it at the source.

`utils/data_loader.py:135` passes the ERA5 file count as the tqdm `total` when loading CHIRPS
with `reproject_to_era5=True`, so the progress bar reads wrong. Cosmetic only.

`archive/cycle_gan_stabilized_experiment.ipynb` references an undefined `LEARNING_RATE` in its
second-stage cell, since the notebook defines `LEARNING_RATE_G` and `LEARNING_RATE_D` instead.
That cell raises `NameError` on a clean run, and the second GAN stage never finished training.

The GAN metrics are quantised, as described in [Results](#results).

## References

1. A. Saha and S. Ravela. "Statistical-Physical Adversarial Learning From Data and Models for
   Downscaling Rainfall Extremes." *Journal of Advances in Modeling Earth Systems* 16(6), 2024.
   [doi:10.1029/2023MS003860](https://doi.org/10.1029/2023MS003860)
2. A. Saha and S. Ravela. "Rapid Climate Model Downscaling to Assess Risk of Extreme Rainfall
   in Bangladesh in a Warming Climate." 2024. [arXiv:2412.16407](https://arxiv.org/abs/2412.16407)
3. A. Saha and S. Ravela. "Rapid Statistical-Physical Adversarial Downscaling Reveals
   Bangladesh's Rising Rainfall Risk in a Warming Climate." 2024.
   [arXiv:2408.11790](https://arxiv.org/abs/2408.11790)
4. J.-Y. Zhu, T. Park, P. Isola, A. A. Efros. "Unpaired Image-to-Image Translation using
   Cycle-Consistent Adversarial Networks." *ICCV*, 2017.
5. P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros. "Image-to-Image Translation with Conditional
   Adversarial Networks." *CVPR*, 2017.
6. E. Schönfeld, B. Schiele, A. Khoreva. "A U-Net Based Discriminator for Generative
   Adversarial Networks." 2021. [arXiv:2002.12655](https://arxiv.org/abs/2002.12655)

Data: CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data, UCSB) and ERA5
(ECMWF), both accessed through Google Earth Engine.

---

Coursework project for EE708. Released under the [MIT License](LICENSE).
