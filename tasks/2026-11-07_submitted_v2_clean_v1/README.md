# CIFAR-10: submitted version 2

**Zihan Xu | NYU Tandon Deep Learning, Fall 2026**

This is the model used for the second Kaggle submission: **87.90% public accuracy (0.87900)**, **2,436,346 trainable parameters**. The submitted CSV was downloaded from Kaggle and matched byte for byte. The private leaderboard score is not recorded.

[Project report](report.pdf) · [Executed review notebook](review.ipynb) · [Submitted CSV](submissions/01_RECOMMENDED_full50k_tta/submission.csv)

The final model is a 3x widened ResNet-20 trained from scratch on all 50,000 permitted images for 240 epochs, seed 42. It uses BlurPool, Cutout, small rotations, dropout and stochastic depth. Inference averages probabilities from the original image and its horizontal mirror. It uses one checkpoint, no external training data or pretrained weights.

## Install

Recorded training environment: Python 3.13.5, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, Pillow 11.1.0, NVIDIA RTX 5070 Ti.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Use a CPU PyTorch installation for prediction and notebook review if CUDA is unavailable. Exact training reproduction requires a CUDA GPU supporting BF16. Run Python normally, without `-O`, because integrity checks use assertions.

## Reproduce the submitted predictions

Obtain the authorized course data separately. Place `train_images/`, `test_images/`, `train_labels.csv`, `test.csv` and `sample_submission.csv` directly inside `data/competition/`. No course images are included here.

Run these commands from this directory:

```bash
python predict.py --data-dir data/competition --output reproduced_tta.csv --compare submissions/01_RECOMMENDED_full50k_tta/submission.csv
python predict.py --data-dir data/competition --policy raw --output reproduced_raw.csv --compare submissions/02_ALTERNATIVE_full50k_raw/submission.csv
python verify.py --data-dir data/competition
```

CPU is the default; add `--device cuda` if desired. Outputs must not already exist. The recommended TTA file is the exact second submitted version. The raw alternative uses the same checkpoint, differs on 69 of 2,000 labels, and has no recorded Kaggle score. Both files contain exactly `id,label`, unique sample-ordered IDs and integer labels 0-9. Their hashes and policies are in [submissions/versions.json](submissions/versions.json).

## Reproduce training

```bash
python train.py --data-dir data/competition --output-dir ../retraining-v2 --check-only
python train.py --data-dir data/competition --output-dir ../retraining-v2
```

The preflight reads all 50,000 original PNGs, verifies their ordered tensor and label hashes, recomputes normalization and checks all 240 sampler schedules. Training processes every image once per epoch, including the final batch of 80: 391 updates per epoch, 93,840 total. The shuffled sampler repairs missing classes by swapping examples so each batch contains all ten classes, without duplicating or dropping examples.

The exact configuration is [config.json](config.json): SGD, LR 0.1, momentum 0.9, weight decay 0.0005, batch 128, cosine decay, no warmup. Flip, rotation and Cutout each activate with probability 0.5. Rotation is uniform within +/-10 degrees with bilinear interpolation and reflection padding. Cutout masks a 12x12 region, clipped at image boundaries, using normalized zero. There is no random crop or Mixup in this submitted version.

Weights are saved after the final prescribed epoch, without validation-based early stopping. Rerun the same command to resume the local `last.pt`, including optimizer, scheduler and random states. Resume files are trusted local files, not downloaded third-party checkpoints. A `control.json` with `{"pause": true}` or `{"cancel": true}` in the output directory stops at an epoch boundary. The output lock prevents duplicate writers to that directory. `--stop-after-epoch 1` supports a short reproduction check; it does not export a completed submission model. Outputs must be outside this release directory.

## Results and evaluation

| Measurement | Result | Scope |
|---|---:|---|
| Kaggle public accuracy | **87.90%** | Actual second submission, full-50k model |
| Earlier Kaggle submission | 85.00% | Earlier core240/raw model, user-confirmed file |
| Research validation, TTA | 96.04% mean; 0.151 percentage-point SD | Three 40k-trained models, seeds 142-144, shared 5k validation set |
| Research validation, raw | 95.9667% mean | Same checkpoints and validation examples |
| Trainable parameters | **2,436,346** | Includes the classifier; below five million |

The full-50k model has **no independent local holdout** because the former validation and local-test examples were included in its final fit. The 96.04% research figure is not its test accuracy. The public Kaggle gain over the earlier submission is 2.90 percentage points; multiple changes occurred together, so this is not a controlled estimate of any individual method's effect.

Thirty confirmation runs (10 configurations x 3 seeds) were completed before selection. All 60 raw/TTA prediction tables are included. Recomputing their counts does not need the training images. The best checkpoint of each research run was selected using raw validation accuracy; TTA was evaluated on that same checkpoint. None of the 34 planned statistical comparisons passed Holm correction at 0.05. Selection ranking is not proof of general superiority.

For a genuinely independent labeled dataset:

```bash
python evaluate.py --checkpoint model.pt --labels independent_labels.csv --images independent_images --policy tta
```

The label CSV needs `filename,label`. The caller must establish independence; a filename guard alone cannot establish provenance. Kaggle hidden labels are unavailable and are not inferred by this command.

## Review and repository contents

```bash
python verify.py
python -m pip install -r requirements-review.txt
python -m jupyter nbconvert --execute --to notebook --inplace review.ipynb
```

The checked-in notebook is already executed. It prints the final public accuracy and parameter count, verifies all research counts and submission hashes, and displays comparison and training plots. Re-execution changes the notebook, so the immutable hash manifest should be checked before rerunning it.

| Path | Purpose |
|---|---|
| `model.py`, `model.pt` | Submitted architecture and unchanged final weights |
| `training.py` | Sampling, normalization, augmentation and loss |
| `train.py`, `trainer.py`, `runtime.py` | Preflight, complete training and safe local checkpointing |
| `predict.py`, `evaluate.py`, `verify.py` | Prediction, independent evaluation and evidence audit |
| `config.json`, `requirements*.txt` | Exact training configuration and dependencies |
| `report.pdf`, `review.ipynb`, `figures/` | Course report and executed visual review |
| `evidence/` | Original completed training/research records and new verification |
| `submissions/` | Two validated policies from this same final model |
| `report_source.py`, `REFERENCES.md` | Rebuildable report and method citations |

The clean code was checked against the original initialization, forward/backward computation, augmentation and first complete training epoch. CPU replay reproduced every submitted prediction. The 240-epoch training run itself was not repeated during cleanup. Historical research entry points and earlier deliveries remain in the repository's [main archive branch](https://github.com/Zihan-Xu/DL-project1/tree/main); this standalone branch is the submitted model, not the later third-version ensemble. The original 30-run training workflow is retained in [the archived v2 instructions](https://github.com/Zihan-Xu/DL-project1/tree/main/tasks/2026-11-06_long111_full50k_v2#reproduce-research-comparisons).

See [REFERENCES.md](REFERENCES.md) for cited methods and [evidence/README.md](evidence/README.md) for provenance. The release manifest covers the exact delivered files.
