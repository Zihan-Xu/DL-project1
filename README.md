# CIFAR-10 residual-network project

**Author and Kaggle team: Zihan Xu.** Course: NYU Tandon Deep Learning, Fall 2026.

This repository contains a reproducible, versioned release of a residual CNN trained from scratch on the course data. The model has **2,436,346 trainable parameters**. The completed 86-run combination study selected `core_ep240` by mean validation accuracy over seeds 42, 43 and 44: **95.3867%** with original-image inference. A subsequent three-seed validation audit of horizontal-flip probability averaging obtained **95.7667%**. Deployment keeps the preselected seed 42; its flip result is **95.78% (4,789 / 5,000)**.

These are local validation results. The user reported **0.85000** on Kaggle for the seed42/raw CSV, and supplied CSV bytes were matched locally. New flip CSVs have no reported Kaggle score. Hidden/private accuracy is unavailable. This release does not claim a leaderboard improvement or a globally optimal configuration.

## Current release

Open [the version directory](tasks/2026-11-06_core_ep240_v1/) for the report, executed notebook, three checkpoints, four CSV choices, exact configurations, 240-epoch logs, comparison table and verification evidence. Use [the recommended CSV](tasks/2026-11-06_core_ep240_v1/submissions/01_recommended_seed42_flip/submission.csv) for seed42 horizontal-flip inference. The other files are alternatives, not an ensemble. Keep the filename `submission.csv` when submitting. All contain exactly `id,label` and 2,000 predictions in official sample order.

New releases are added under `tasks/<date>_<task>_<version>/`; existing snapshots are retained. Results from incomplete or unaccepted studies are not promoted.

## Environment and data

Verified training environment: Windows, Python 3.13.5, PyTorch 2.11.0+cu128, NumPy 2.1.3, Pillow 11.1.0, NVIDIA RTX 5070 Ti. Training uses CUDA/BF16 and FP32 validation. Prediction and notebook review can run on CPU. Versions and hardware can affect floating-point results; compare predictions using the supplied command.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For CPU-only prediction install the CPU build of PyTorch instead. Place authorized course files in `data/competition/`: `train_images/`, `test_images/`, `train_labels.csv`, `test.csv`, `sample_submission.csv`. Raw images and caches are not redistributed. The fixed split is 40,000 training / 5,000 validation / 5,000 historical local test, stratified by class with seed42. Normalization uses training images only. Competition images are used only for inference.

## Reproduce a CSV

Run from the repository root after supplying course data. The script reads every original competition PNG and refuses to overwrite an existing output.

```powershell
python predict.py --data-dir data/competition --checkpoint tasks/2026-11-06_core_ep240_v1/models/seed42.pt --policy flip --output reproduced.csv --compare tasks/2026-11-06_core_ep240_v1/submissions/01_recommended_seed42_flip/submission.csv
```

For the original inference use `--policy raw` and compare against `02_reference_seed42_raw/submission.csv`. For alternatives use seed43 or seed44 and the matching version. Flip inference averages softmax probabilities of the original image and its horizontal mirror from **one checkpoint**, then takes argmax. It doubles forward-pass cost without adding parameters. It does not combine different models or use 90-degree rotations.

## Reproduce training

The original training, augmentation, sampler, scheduler and model modules are included unchanged under `experiments/research_sweep/`. `train.py` selects a single released configuration, bypassing the full historical sweep while calling its original training implementation. It does not require a teacher or any unpublished model. The recorded source/config hashes are retained for provenance.

```powershell
python prepare_data.py
python train.py --config tasks/2026-11-06_core_ep240_v1/evidence/seed42/config.json --check-only
python train.py --config tasks/2026-11-06_core_ep240_v1/evidence/seed42/config.json --output-dir retraining/seed42
```

Use the corresponding seed43/44 config for repetitions. `prepare_data.py` audits the supplied original images, regenerates the cache and fixed split, and computes training-only normalization. The training wrapper requires CUDA; it stores resumable checkpoints and fresh evaluation evidence outside the released snapshot. Full 240-epoch retraining was not repeated during packaging; the included original runs and independent replay evidence document those executions. Historical full-sweep commands require additional historical artifacts; use the single-run wrapper above for this release.

After training, the wrapper also writes `retraining/seed42/model.pt`, which can be passed directly to `predict.py --checkpoint` to generate predictions for the newly trained model.

Training: 3x ResNet-20, channels 48/96/192, three two-convolution residual blocks per stage, BlurPool downsampling, random horizontal flip p=.5, Cutout12 p=.5, no random crop, SGD learning rate .1 / momentum .9 / weight decay .0005, batch128, 240 epochs, cosine schedule, ordinary cross-entropy. Each epoch evaluates raw validation images; the largest correct count selects the checkpoint, ties going to the earlier epoch. Inference policy is evaluated separately on the same validation set.

## Inspect results

[Review notebook](tasks/2026-11-06_core_ep240_v1/review.ipynb) includes executed parameter-count, validation, reported Kaggle-score, CSV-integrity and figure cells. [Report](tasks/2026-11-06_core_ep240_v1/report.pdf) records the architecture, methodology, comparisons, limitations and references. The repository link is on its first page. Re-run the notebook from its folder with a Python environment containing `requirements.txt`.

`SHA256SUMS.json` records distributed file digests. Version metadata identifies each CSV's checkpoint, seed, inference policy and score provenance. Evaluation labels from the fixed course-training holdout are distinguished from unavailable hidden competition labels.

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016. https://arxiv.org/abs/1512.03385
- R. Zhang. *Making Convolutional Networks Shift-Invariant Again*. ICML 2019. https://arxiv.org/abs/1904.11486
- T. DeVries, G. W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout*. 2017. https://arxiv.org/abs/1708.04552
- Course-provided *Deep Learning Project 1, Fall 2026* and [competition rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules).

Source implementation is local project code inspired by the cited methods. No external pretrained weights or external training data are included or used for this released model. Further accepted training tasks may supersede this version; the release is a fixed snapshot.
