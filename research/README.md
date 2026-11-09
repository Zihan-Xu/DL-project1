# Reproducing the development experiments

The report describes experiments that preceded the submitted model. This folder supplies their original model, preprocessing, augmentation and training code, together with the fixed project-trained teacher and the 1,000-row manual background review. The submitted model is still the checkpoint at the repository root; use the root README to reproduce its predictions or full-data training.

`SOURCE_SHA256.json` identifies every original file copied here. The training and model files match their project sources byte for byte. English protocol notes and a small English reporting adapter are supplied for fresh workspaces. The adapter recomputes correct counts from prediction tables and writes means and sample standard deviations; it does not replace the original statistical analysis or claim that a rerun has passed the historical independent acceptance checks. The recorded comparisons and acceptance results remain in `../evidence/`.

The original execution environment was Windows, Python 3.13.5, PyTorch 2.11.0+cu128 and one RTX 5070 Ti. Some orchestration code uses Windows file locks. OpenCV 4.12.0.88 is needed to rebuild the background pool. Exact numerical agreement depends on the recorded environment and deterministic settings. These commands perform substantial training; the final delivery review checked the code and inputs but did not retrain the entire research matrix.

## Create isolated workspaces

Install the root requirements and `research/requirements.txt`. Run from the submission repository:

```bash
python research/create_workspace.py --output-dir ../development-rerun --mode studies --data-dir data/competition
python research/create_workspace.py --output-dir ../width-rerun --mode width --data-dir data/competition
```

The optional `--data-dir` copies only the course images and CSVs into the new workspace. Otherwise, place those files in its `data/competition/` folder yourself. The destination must be new and outside this release. No training starts during workspace creation. Keep both workspaces separate: later studies use a fixed historical teacher, while the width sweep trains fresh models.

## Baseline and width sweep

From `width-rerun`, run:

```bash
python prepare_data.py
python run_baseline.py train
python run_baseline.py evaluate
python run_widths.py train --widths 2 3 4
python run_widths.py compare
```

`prepare_data.py` audits the PNGs, generates the fixed stratified split and caches decoded pixels. `config.json` is the initial 80-epoch crop-and-flip baseline. The comparison selects checkpoints by validation accuracy. The historical local-test check is a separate output and must not be used to alter that selection.

## Augmentation and paired fine-tuning

From `development-rerun`, run:

```bash
python prepare_data.py
python run_baseline.py train
python run_baseline.py evaluate
python prepare_backgrounds.py
python run_augmentations.py train A0_none A1_flip A2_rotation A3_color A4_resolution A5_noise A6_background
python run_augmentations.py compare
python run_posttraining.py A_sft
python run_posttraining.py B_supcon
python run_posttraining.py compare
```

The background pool is regenerated from training images with fixed segmentation and donor-selection settings. No original or synthetic course images are distributed here. `experiments/paired_posttraining/pair_review.csv` preserves the actual keep/reject decisions; the paired loader checks source and donor identities before using the retained 750 pairs.

The studies workspace includes the exact 3x width checkpoint used as the paired fine-tuning parent and distillation teacher, plus its width-selection record. It was trained within this project, with SHA-256 `aa64543eba16b580a5a6f6bb8b88d31afeff1156eefcd7bd7b62f8e948b0d02a`. It is a development reference, not the final full-50k submission checkpoint. New width experiments belong in the separate width workspace and do not overwrite it.

## Architecture study: 42 runs

After data preparation in `development-rerun`:

```bash
python experiments/research_sweep/architecture/runner.py prepare
python experiments/research_sweep/architecture/runner.py smoke
python experiments/research_sweep/architecture/durable_runner.py run
```

The executable table defines the twelve configurations and their three- or five-seed schedules. Preparation checks parameter counts, sampling and resume behavior before freezing the new run inputs. The durable wrapper changes checkpoint I/O, not model computation.

## Data study: 71 results

Run the background preparation above first. Then:

```bash
python run_color_balance.py prepare
python run_color_balance.py train
python run_color_balance.py compare
python experiments/research_sweep/data/study.py
python experiments/research_sweep/data/check_prior.py
python experiments/research_sweep/data/runner.py smoke
python experiments/research_sweep/data/runner.py freeze
python experiments/research_sweep/data/runner.py all
```

The first three commands recreate the six label/tone runs. The compatibility check compares their complete sampling and augmentation plans with the wider data study. The runner then imports those six results after replay and trains the other 65. This preserves the historical accounting rather than counting the reused results as new runs.

## Training study: 120 outputs from 107 trajectories

```bash
python experiments/research_sweep/training/register.py
python experiments/research_sweep/training/preflight.py
python experiments/research_sweep/training/runner.py
```

Registration specifies all 34 configurations, including shared trajectories for averaging methods. Preflight tests numerical resume behavior and the frozen project teacher. Filter an exploratory rerun with `--only base --seed 42`; a filtered run does not complete the full study.

## Combinations, inference and final confirmation

```bash
python experiments/research_sweep/combinations/runner.py prepare
python experiments/research_sweep/combinations/runner.py smoke
python experiments/research_sweep/combinations/runner.py run
python delivery_tools/audit_rotation_flip_tta.py
python experiments/rotation90_training/study.py
python experiments/rotation90_training/replay.py
python experiments/research_sweep/confirmation/runner.py prepare
python experiments/research_sweep/confirmation/runner.py smoke
python experiments/research_sweep/confirmation/runner.py run
```

The combination registration defines 22 configurations and 86 runs. The rotation-inference script compares fixed views on the three saved `core_ep240` checkpoints. The right-angle training study reuses the three `core_111` controls and trains three rotation variants. Confirmation defines ten configurations with seeds 142-144; its runner also evaluates the fixed horizontal-flip policy on the selected checkpoints.

The English adapter writes `reproduction_runs.csv`, `reproduction_summary.csv` and `REPORT.md` within each study directory. Original per-image predictions and saved checkpoints remain the basis for reviewing rerun results. Do not mix a new partial run with the recorded historical means or relabel a local result as a Kaggle score.
