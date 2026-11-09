# CIFAR-10 classification with a modified ResNet

**Zihan Xu | NYU Tandon Deep Learning, Fall 2026**

This repository contains the model used for my second Kaggle submission: **87.90% public accuracy**, with **2,436,346 trainable parameters**. It is a 3x widened ResNet-20 trained from scratch on all 50,000 permitted images for 240 epochs. Prediction averages the probabilities from an image and its horizontal mirror using one checkpoint. This is horizontal-flip test-time augmentation (TTA).

[Report (8 pages)](report.pdf) · [Executed notebook](review.ipynb) · [Submitted CSV](submissions/01_RECOMMENDED_full50k_tta/submission.csv) · [References](REFERENCES.md)

## Results at a glance

| Measurement | Result | Evaluation |
| --- | --- | --- |
| Second Kaggle submission | **87.90% public accuracy** | Final model trained on 50,000 images |
| Earlier Kaggle submission | 85.00% public accuracy | Earlier core240 model with raw inference |
| Selected recipe, flip TTA | 96.040% mean; 0.151 percentage-point seed SD | Three models trained on 40,000 images; fixed 5,000-image validation set |
| Same research checkpoints, raw inference | 95.967% mean | Same three seeds and validation images |
| Trainable parameters | 2,436,346 | Includes the classifier; below five million |

Seed SD is the standard deviation across training seeds. Differences labeled pp are measured in percentage points.

The 96.04% validation result is not the final model's test accuracy. The final fit includes the former holdout images, so it has no independent local holdout. The private Kaggle score is not recorded. The 2.90-point public-score improvement followed several changes together, including training-set size and inference policy; it cannot identify one component's contribution.

The downloaded second-submission CSV matched the released file byte for byte. The [score verification record](evidence/kaggle_submission_verified.json) documents that match. The validation/public gap remains unexplained; repeated validation selection and differences between evaluation datasets are possible explanations, not established findings.

## Run the submitted model

The recorded environment is Python 3.13.5, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, Pillow 11.1.0 and one NVIDIA RTX 5070 Ti.

```bash
python -m venv .venv
# Activate the environment in your shell, then install:
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

A CPU build of PyTorch is sufficient for prediction and notebook review. Reproducing training requires a CUDA GPU with BF16 support. Run Python without `-O`, because verification uses assertions.

Obtain the course data separately and place `train_images/`, `test_images/`, `train_labels.csv`, `test.csv` and `sample_submission.csv` inside `data/competition/`. Run from the repository root:

```bash
python predict.py --data-dir data/competition --output reproduced_tta.csv --compare submissions/01_RECOMMENDED_full50k_tta/submission.csv
python predict.py --data-dir data/competition --policy raw --output reproduced_raw.csv --compare submissions/02_ALTERNATIVE_full50k_raw/submission.csv
python verify.py --data-dir data/competition
```

Prediction uses CPU by default; add `--device cuda` to use the GPU. Output files must not already exist. Both CSV versions have exactly `id,label`, 2,000 unique IDs in sample order and integer labels 0-9. The raw alternative uses the same checkpoint, differs on 69 labels and has no recorded Kaggle score. [versions.json](submissions/versions.json) records the policies and hashes.

## How the final model was chosen

### Data checks and baseline

The input audit checked image decoding, IDs, labels and pixel hashes. The data contain 50,000 labeled and 2,000 competition images, all 32 x 32 RGB. No exact duplicates, conflicting duplicate labels or exact training/competition overlaps were found. A fixed stratified split used seed 42 and 40,000/5,000/5,000 images for training, validation and a historical local test: 4,000/500/500 per class. Normalization used training pixels only.

The initial ResNet-20 used stage widths 16/32/64, three residual blocks per stage, a 3 x 3 stem, global average pooling and a ten-class classifier. It trained for 80 epochs with SGD, learning rate 0.1, momentum 0.9, weight decay 0.0001, batch size 128 and cosine decay. Reflection-padded random cropping and horizontal flipping were used. The selected epoch-75 checkpoint reached 90.94% validation accuracy and 90.66% on the historical local test.

### Width and initial augmentation tests

The width sweep changed only the stage widths under the initial crop-and-flip recipe. Each model used seed 42 and 80 epochs.

| Width | Stage channels | Parameters | Validation accuracy (%) |
| --- | --- | --- | --- |
| 1x | 16/32/64 | 272,474 | 90.94 |
| 2x | 32/64/128 | 1,084,586 | 92.76 |
| 3x | 48/96/192 | 2,436,346 | 93.44 |
| 4x | 64/128/256 | 4,327,754 | 93.20 |

I kept the 3x model. The 4x model used about 78% more multiply-accumulates per image and did not improve validation accuracy for this seed. The selected 3x checkpoint reached 93.52% on the historical local test. This single-seed comparison guided the initial capacity choice; later studies tested the recipe across multiple seeds. [Width results](evidence/development/width.csv).

The next study tested individual augmentations on the narrow model. These arms removed random cropping, used seed 42 and trained for 80 epochs. Each transformation was applied with probability 0.5; the control used normalization alone.

| Augmentation | Validation (%) | Change from none (pp) |
| --- | --- | --- |
| None | 83.84 | +0.00 |
| Horizontal flip | 86.76 | +2.92 |
| Rotation within +/-10 degrees | 86.30 | +2.46 |
| Brightness, contrast and saturation | 83.72 | -0.12 |
| Resize 32 to 28 to 32 | 82.46 | -1.38 |
| Gaussian noise, SD 0.02 | 83.74 | -0.10 |
| Background replacement | 83.28 | -0.56 |

Small rotations and horizontal flips helped in this setup. Color factors in [0.9, 1.1], reduced resolution and the tested noise level did not. The comparison is against no augmentation, not against the earlier crop-and-flip baseline. [Augmentation results](evidence/development/augmentation.csv).

Background replacement used GrabCut masks, donor-background inpainting and a second mask check with intersection-over-union of at least 0.75. It produced 7,710 accepted training pairs. Visual review retained 750 of a 1,000-pair sample for a separate fine-tuning test. Ten further epochs of cross-entropy reached 93.28% validation accuracy; adding a pair-weighted supervised contrastive loss reached 93.10%. Both were below the 93.44% parent checkpoint. I retained the parent, despite a slightly higher historical local-test score for one fine-tuned arm. [Paired fine-tuning results](evidence/development/paired_posttraining.csv).

### Systematic studies

The next phase covered 34 research directions. Its default was the 3x model, horizontal flipping and 80 epochs, without random cropping. Configurations used seeds 42-44, with seeds 45-46 added for comparisons designated in advance. Each study had its own matched controls and normalization protocol. Results across different protocols should not be interpreted as one controlled progression.

| Study | Completed work | Findings used in the next phase |
| --- | --- | --- |
| Architecture: 8 directions | 42 runs | BlurPool improved its matched control by 2.47 points over three seeds; fixed-width extra depth did not help |
| Data: 12 directions | 71 results: 65 new runs and 6 verified historical imports | Cutout 12 improved all five seeds by a mean 1.288 points; fixed small subsets lost accuracy |
| Training: 14 directions | 120 outputs from 107 training trajectories | Weight decay 0.0005 gave a five-seed paired gain of 1.896 points; longer training and modest regularization warranted combination tests |

Architecture variants included depth/width tradeoffs, channel allocation, pre-activation, channel attention, grouped convolutions and BlurPool. BlurPool added no trainable parameters but increased theoretical computation by 19.6% and measured training time by about 55% against its matched control. It was useful, but not free. [Architecture summary](evidence/development/architecture_summary.csv) and [paired comparisons](evidence/development/architecture_comparisons.csv).

Data experiments also tested sample coverage, class balance, color balance, grayscale and background quality. At equal image-presentation budgets, fixed 20,000- and 10,000-image pools lost 4.71 and 11.37 points against their full-data controls. Rotating the 20,000-image selection recovered 4.45 points over the fixed pool. Strictly equal class counts in every batch reduced accuracy by 0.644 points. Tone balancing and filtered background additions did not show a clear gain. These results favored retaining full data and testing Cutout in the combined model. [Data summary](evidence/development/data_summary.csv) and [paired comparisons](evidence/development/data_comparisons.csv).

Training experiments varied epoch budget, learning rate, schedule, weight decay, optimizer, batch size, label smoothing, dropout and residual dropping. They also tested sharpness-aware minimization, weight averaging, supervised contrastive learning and distillation. Weight decay 0.0003, weight decay 0.0005 and temperature-4 distillation passed Holm correction among the 32 planned comparisons. Distillation improved a smaller 2x student, not the submitted 3x model. Several averaging and more expensive optimization methods had uncertain gains; I kept ordinary SGD. [Training summary](evidence/development/training_summary.csv) and [paired comparisons](evidence/development/training_comparisons.csv).

The 120 training outputs are not 120 independent training trajectories: some averaging methods shared a trajectory. The data count likewise includes six reused results whose compatibility was checked before inclusion. Completion records and per-seed metrics are in [development evidence](evidence/development/).

### Combination tests and confirmation

The combination study completed 86 runs across 22 configurations. Its central five-seed factorial tested every combination of BlurPool, Cutout 12 and weight decay 0.0005. The full core averaged 94.648%, versus 90.340% for its matched base: +4.308 points, paired 95% interval [3.807, 4.809]. Extensions tested rotation, dropout, residual dropping, longer training and other candidates. This tested combinations directly instead of adding gains from separate studies. [Combination results](evidence/development/combination_summary.csv).

On three 240-epoch core checkpoints, horizontal-flip TTA raised the mean from 95.387% to 95.767%. Adding small rotated inference views did not beat flip-only inference. A separate right-angle rotation training test also failed to improve its control. [Inference comparison](evidence/development/rotation_flip_tta.json) and [right-angle training results](evidence/development/rotation90_training.csv).

The final confirmation study used new training seeds 142-144 and completed 30 runs. All models used the core recipe. The three binary factors below are rotation within +/-10 degrees, dropout 0.1 and maximum residual drop 0.1; 1 enables a factor and 0 disables it.

| Rotation / dropout / residual drop | Epochs | Raw mean (%) | Flip TTA mean (%) |
| --- | --- | --- | --- |
| 000 | 80 | 94.827 | 95.233 |
| 001 | 80 | 95.047 | 95.387 |
| 010 | 80 | 95.080 | 95.240 |
| 011 | 80 | 94.927 | 95.267 |
| 100 | 80 | 95.173 | 95.527 |
| 101 | 80 | 95.313 | 95.533 |
| 110 | 80 | 95.233 | 95.467 |
| 111 | 80 | 95.440 | 95.660 |
| 000 | 240 | 95.427 | 95.733 |
| 111 | 240 | 95.967 | 96.040 |

Raw and TTA evaluation used the same checkpoint, selected by raw validation accuracy with earlier epochs winning ties. Selection maximized the total correct count across all three seeds and used a fixed tie rule: raw inference, then fewer epochs, then configuration name. It selected `long_111` with TTA. The seed accuracies were 96.06%, 95.88% and 96.18%.

TTA added only 0.073 points for the selected recipe, with a paired 95% interval of [-0.396, 0.543]. None of the 34 confirmation comparisons passed Holm correction at 0.05. The highest observed mean determined selection, but it does not prove that every component gives a reliable benefit. New seeds measure training variability on the same images, not performance on a new dataset. The [confirmation report](evidence/confirmation_REPORT.md) includes every configuration, inference policy and planned comparison.

## Final architecture and training recipe

The submitted model has nine residual blocks with stage widths 48/96/192. At stage transitions, a fixed 3 x 3 binomial BlurPool filter downsamples both the residual and shortcut paths; a learned 1 x 1 projection matches shortcut channels. Dropout is applied before the second convolution. Per-sample residual dropping increases from 0.1/9 to 0.1 across the blocks, with surviving residuals rescaled during training. Masking happens after residual computation, so it does not skip convolutional work.

| Setting | Submitted value |
| --- | --- |
| Configuration | `long_111`; one checkpoint |
| Data and seed | All 50,000 permitted labeled images; seed 42 |
| Training budget | 240 epochs; 93,840 optimizer updates |
| Optimizer | SGD, learning rate 0.1, momentum 0.9, weight decay 0.0005 |
| Learning-rate schedule | Cosine decay; no warmup or restarts |
| Batch size | 128, with a final batch of 80 |
| Augmentation | Horizontal flip, rotation within +/-10 degrees, Cutout 12; each probability 0.5 |
| Regularization | Dropout 0.1; maximum residual drop 0.1 |
| Precision | BF16 training; FP32 prediction |
| Inference | Equal average of original and mirrored softmax probabilities |

Rotation uses bilinear interpolation and reflection padding. Cutout fills a 12 x 12 region with normalized zero, clipped at image boundaries. The shuffled sampler swaps examples to ensure each batch contains all ten classes without forcing equal counts, duplicating examples or dropping the last batch. Every image appears once per epoch. This submitted version uses no random crop, Mixup, external training data or pretrained weights.

Normalization was recomputed for the full-data fit; exact means and standard deviations are in [config.json](config.json). The final checkpoint is epoch 240, without validation-based early stopping. Two-pass TTA costs about 872.2 million multiply-accumulates per image and adds no parameters.

## Reproduce training and review the results

```bash
python train.py --data-dir data/competition --output-dir ../retraining-v2 --check-only
python train.py --data-dir data/competition --output-dir ../retraining-v2
```

Preflight reads all 50,000 training PNGs, checks ordered image/label hashes, recomputes normalization and verifies all 240 sampler schedules. Rerunning the training command resumes the local `last.pt`, including optimizer, scheduler and random states. Use a fresh output directory outside the release. The output lock prevents two processes writing to the same run.

For a short reproduction check, add `--stop-after-epoch 1`; this does not export a completed submission model. A `control.json` containing `{"pause": true}` or `{"cancel": true}` in the run directory stops at an epoch boundary.

For a separate labeled dataset that is independent of training:

```bash
python evaluate.py --checkpoint model.pt --labels independent_labels.csv --images independent_images --policy tta
```

The labels file needs `filename,label`. Dataset independence must be established from its source; the command cannot recover Kaggle's hidden labels.

The executed notebook checks the final parameter count, 30 confirmation runs, 60 prediction tables, both submission hashes and all four development-study summary tables. It also plots confirmation results and the final training curve. To review locally:

```bash
python verify.py
python -m pip install -r requirements-review.txt
python -m jupyter nbconvert --execute --to notebook --inplace review.ipynb
```

Run the integrity check before re-executing the notebook, because execution changes its file hash. The code cleanup was checked against the original initialization, forward/backward computation, augmentation and first complete training epoch. CPU replay reproduced all raw and TTA labels. The full 240-epoch fit was not repeated during cleanup.

The report contains seven tables, one training figure and references within eight pages. Rebuild it with:

```bash
python -m pip install -r requirements-report.txt
python report_source.py --data-root . --output rebuilt_report.pdf
```

## Repository contents

| Path | Contents |
| --- | --- |
| `model.py`, `model.pt` | Submitted architecture and trained weights |
| `training.py` | Sampling, normalization, augmentation and loss |
| `train.py`, `trainer.py`, `runtime.py` | Training, preflight and checkpoint/resume handling |
| `predict.py`, `evaluate.py`, `verify.py` | Prediction, independent evaluation and artifact checks |
| `config.json`, `requirements*.txt` | Exact configuration and dependencies |
| `report.pdf`, `report_source.py`, `review.ipynb` | Report, its builder and executed review |
| `evidence/development/` | Earlier study summaries, per-seed metrics and acceptance records |
| `evidence/research_logs/` | Confirmation training logs and all 60 prediction tables |
| `submissions/` | The submitted TTA CSV and raw alternative, with hashes and policies |

The default `submission-v2` branch is the course delivery. Earlier releases and research are preserved on the [main archive branch](https://github.com/Zihan-Xu/DL-project1/tree/main), including the [original confirmation training workflow](https://github.com/Zihan-Xu/DL-project1/tree/main/tasks/2026-11-06_long111_full50k_v2#reproduce-research-comparisons). The later third-version experiment is separate from this submitted model.

[REFERENCES.md](REFERENCES.md) follows the report's citation numbering. [evidence/README.md](evidence/README.md) explains score provenance, frozen results and the English editions of historical text. The release manifest records the delivered file hashes.
