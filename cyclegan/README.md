# CycleGAN downscaling

CycleGAN experiments were run against the reference implementation from
[`junyanz/pytorch-CycleGAN-and-pix2pix`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
rather than reimplemented. This directory holds only the files that are ours — the custom
dataset class, the driver notebook, and a patch against upstream. The ~100k lines of upstream
code are not vendored here.

A from-scratch reimplementation of the same idea, self-contained and runnable without the
upstream repo, lives in [`../07_cyclegan_stabilized.ipynb`](../07_cyclegan_stabilized.ipynb).

## Contents

| File | Installs to | Purpose |
|---|---|---|
| `custom_train.ipynb` | repo root | Driver notebook. Shells out to `train.py` / `test.py` via `subprocess`, then plots results, Q–Q bias, and loss curves. |
| `pickle_dataset.py` | `data/pickle_dataset.py` | `PickleDataset(BaseDataset)`, registered by filename convention as `--dataset_mode pickle`. Adds `--pickle_file_A` / `--pickle_file_B`. |
| `upstream_patches.diff` | applied at root | Edits to `options/base_options.py`, `util/util.py`, `util/visualizer.py`, `test.py`. |
| `show_checkpoint_images.ipynb` | repo root | Plots the sample grids written to `checkpoints/<name>/web/images/`. |
| `experiment.py` | repo root | Scratch stub: builds options + dataset + model and stops. Not part of the pipeline. |

What `upstream_patches.diff` changes:

- `options/base_options.py` — makes `--dataroot` optional (paths come from `--pickle_file_A/B`
  instead) and adds `pickle` to the `--dataset_mode` help string.
- `util/util.py` — adds `tensor2im_custom()`, a grayscale-aware `tensor2im` that returns
  `H×W` instead of tiling to RGB, plus `save_pickle_image()` / `save_pickle_images()`.
- `util/visualizer.py` — adds `save_images_custom()`, which stacks visuals as float64 and
  pickles them instead of writing 8-bit PNGs.
- `test.py` — switches to `save_images_custom`. Note that in the patched state the original
  `save_images(...)` and `webpage.save()` calls are **commented out**, so `test.py` writes
  nothing until you re-enable one of the two paths.

## Setup

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
git checkout 2a7afba          # upstream tip this work was based on

git apply /path/to/cyclegan/upstream_patches.diff
cp /path/to/cyclegan/pickle_dataset.py            data/pickle_dataset.py
cp /path/to/cyclegan/custom_train.ipynb           .
cp /path/to/cyclegan/show_checkpoint_images.ipynb .
```

## Data

The four inputs are byte-identical copies (md5-verified) of pickles produced by this project's
notebooks `04_data_preparation.ipynb` and `05_downscale_chirps.ipynb`:

| Destination in the CycleGAN repo | Source in `../data/` | Entries | Value shape |
|---|---|---|---|
| `datasets/era52chirps/trainA/era5.pkl` | `final_era5.pkl` | 519 | `(5, 64, 64)` |
| `datasets/era52chirps/trainB/chirps.pkl` | `final_chirps.pkl` | 519 | `(64, 64)` |
| `datasets/chirps_upscaled_2_chirps/trainA/chirps_reprojected.pkl` | `filtered_chirps_reprojected.pkl` | 682 | `(64, 64)` |
| `datasets/chirps_upscaled_2_chirps/trainB/chirps.pkl` | `filtered_chirps.pkl` | 682 | `(64, 64)` |

Each is a `dict[datetime.date, np.ndarray(float32)]`, keys starting mid-1999. Domain A of
`era52chirps` carries the 5 stacked ERA5 channels (air temperature, dewpoint, precipitation,
u-wind, v-wind); everything else is single-channel precipitation in mm/day.

## Experiments

Shared flags for both runs:

```
--model cycle_gan --dataset_mode pickle
--netG resnet_9blocks --netD n_layers --n_layers_D 4 --norm instance
--input_nc 1 --output_nc 1 --ngf 64 --ndf 64
--batch_size 8 --preprocess none
--n_epochs 50 --n_epochs_decay 50
```

`--preprocess none` makes `--load_size` / `--crop_size` inert; the 64×64 arrays pass through
untouched. Upstream defaults left in place: `lambda_identity 0.5`, `gan_mode lsgan`,
`pool_size 50`, `beta1 0.5`, `direction AtoB`, `init_type normal`, `save_epoch_freq 5`.

**1. `era52chirps`** — ERA5 → CHIRPS, the real downscaling task.
`--lambda_A 40 --lambda_B 40 --lr 0.00002 --use_wandb`, 519 training images.

**2. `chirps_upscaled_2_chirps`** — degraded CHIRPS → CHIRPS, the idealised task.
`--lambda_A 10 --lambda_B 10`, default `--lr 0.0002`, 682 training images.

Model sizes: G_A and G_B are 11.366 M parameters each, D_A and D_B 6.958 M each.

## Results

From `custom_train.ipynb` cell 18, on the `chirps_upscaled_2_chirps` test outputs:

```
Mean Bias: 56.353
RMSE:      6.667
```

These are computed on the 8-bit 0–255 greyscale PNGs that `test.py` writes, **not** on mm/day
values, so they are not directly comparable to the mm/day metrics reported for the linear
regression baseline in the top-level README. A mean bias larger than the RMSE is the visible
symptom of that.

Best `era52chirps` losses at epoch 100: `D_A 0.003, G_A 0.867, cycle_A 0.311, idt_A 0.105,
D_B 0.054, G_B 0.632, cycle_B 0.348, idt_B 0.067`.

## Known issues

- **`era52chirps` checkpoints are not preserved.** That run completed 100/100 epochs on a
  remote GPU pod (wandb run `ie81bss0`, 2025-11-10) but only the
  `chirps_upscaled_2_chirps` generator weights were pulled back. Reproducing the ERA5→CHIRPS
  numbers means retraining.
- **`chirps_upscaled_2_chirps` is unstable.** It reached epoch 67 before being stopped, and
  the run captured in the notebook outputs goes `nan` on every loss from iteration 200
  onward, with `RuntimeWarning: invalid value encountered in cast` from `util/util.py`.
- **`pickle_dataset.py` is single-channel only as shipped.** It passes the whole array to
  `Image.fromarray`. For `era52chirps`, whose domain A is `(5, 64, 64)`, restore the
  channel selection:
  ```python
  A_img = Image.fromarray(A_image[2])   # index 2 = total precipitation
  ```
- **Batch size drift.** All eight recorded wandb runs used `--batch_size 1` despite the
  notebook constant being 8; the saved cell output for `era52chirps` is from one of the
  batch-1 runs.
- `custom_train.ipynb` cell 20 reads `checkpoints/<name>/loss_log.txt`, and
  `show_checkpoint_images.ipynb` reads `checkpoints/<name>/web/images/`. Neither exists
  unless you train locally.
