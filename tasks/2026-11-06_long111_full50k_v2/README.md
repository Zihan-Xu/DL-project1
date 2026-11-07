# Final full-50k release: long_111 + horizontal-flip TTA

Author and Kaggle team: **Zihan Xu**. NYU Tandon Deep Learning, Fall 2026.

**Recommended:** `submissions/01_RECOMMENDED_full50k_tta/submission.csv`.
Only upload this CSV to Kaggle. Four version folders contain valid, distinct CSVs and metadata.
The raw full50k alternative uses the same weights with one forward pass. Previous core240 files are retained for reference. No leaderboard score is available for the new model; 0.85000 is the user's reported score for the previous raw CSV, not for this release.

## Findings and scope

The accepted confirmation study completed 30/30 runs (10 configurations x seeds142-144), followed by raw and horizontal-flip inference on every fixed checkpoint. `long_111/tta` ranked first among the 20 measured configuration/policy choices: 96.0400% mean accuracy, SD 0.1510 percentage points, on the same 5,000-image research validation set. It has 2,436,346 trainable parameters. Raw mean was 95.9667%; the TTA gain was only 0.0733 percentage points and was uncertain across three seeds. No one of 34 comparisons passed Holm correction at 0.05.

After selection, seed42 was fixed and a new model was trained from scratch on all 50,000 course training images for 240 epochs. Final weights are from epoch240, without early stopping. RGB mean/std were recomputed on all 50k. The original validation and historical local-test images are now training data: **there is no independent local test accuracy for this final model**. The research mean is not its test accuracy, nor a promised Kaggle score.

## Environment and data

Use Python3.13.5, PyTorch2.11.0+cu128, NumPy2.1.3 and Pillow11.1.0 for the recorded environment (RTX5070Ti). The repository root README gives installation commands. `requirements.txt` also includes notebook packages. Training requires CUDA/BF16; prediction/review run on CPU. Hardware/library differences can change floating-point results.

Supply authorized data in repository-root `data/competition`: `train_images/`, `test_images/`, `train_labels.csv`, `test.csv`, `sample_submission.csv`. Raw images, external datasets, pretrained weights and test-label annotations are not distributed or used. The independent training preflight verifies the original training CSV and all decoded image bytes against the recorded hashes.

## Predict the recommended CSV

All commands below run from the repository root:

```powershell
python tasks/2026-11-06_long111_full50k_v2/code/predict.py --data-dir data/competition --output reproduced_tta.csv --compare tasks/2026-11-06_long111_full50k_v2/submissions/01_RECOMMENDED_full50k_tta/submission.csv
python tasks/2026-11-06_long111_full50k_v2/code/predict.py --data-dir data/competition --policy raw --output reproduced_raw.csv --compare tasks/2026-11-06_long111_full50k_v2/submissions/02_ALTERNATIVE_full50k_raw/submission.csv
```

TTA averages the softmax probabilities from the original image and its horizontal mirror with equal weights, then takes argmax. It uses one model, two FP32 passes, no rotation/crop search. The final checkpoint defaults to this release's `model.pt`. Outputs must not already exist. The two old CSVs use the repository-root `predict.py` and the v1 model; see the v1 README.

## Reproduce full50k training

```powershell
python tasks/2026-11-06_long111_full50k_v2/code/train_full50k.py --data-dir data/competition --output-dir retraining/full50k --check-only
python tasks/2026-11-06_long111_full50k_v2/code/train_full50k.py --data-dir data/competition --output-dir retraining/full50k
```

The adapter uses the actual executed training loop in `code/engine.py`; only its runtime import is changed. It replaces project queue/acceptance discovery and cache preparation with standalone original-PNG preparation, fixed accepted configuration, and an output-directory lock. `source_archive/train_executed.py` preserves the original entry point. Architecture and operators are byte-identical to the trained source. Use a free GPU: the standalone lock prevents duplicate writers in one output directory, but does not coordinate other GPU processes. Resume by rerunning the same command: optimizer, scheduler and all RNG states are restored from `last.pt`. If a crash interrupts atomic file rotation, preserve the files and inspect `previous.pt` before recovery. A `control.json` in the output folder with `{"pause": true}` or `{"cancel": true}` stops at an epoch boundary.

Training is SGD lr0.1, momentum0.9, weight decay0.0005, batch128, cosine240, no warmup. Augmentation is horizontal flip p0.5, uniform rotation +/-10 degrees p0.5 with bilinear/reflection sampling, Cutout12 p0.5. The 3x ResNet20 has channels48/96/192, BlurPool downsampling, dropout0.1 before the second convolution, and linearly increasing per-sample residual drop probability up to0.1. There is no teacher, external weight, random crop or attention in the selected recipe. Each epoch processes all 50k once, including the last80-image batch: 391 updates/epoch, 93,840 overall. The final model is not chosen using the original holdout.

## Reproduce research comparisons

The full matrix, all 30 original run configurations, epoch logs, raw and TTA per-image predictions are included in `evidence/`. Recompute every correct count and mean by executing `review.ipynb`. For a fresh original-loop training repetition:

```powershell
python prepare_data.py
python tasks/2026-11-06_long111_full50k_v2/code/train_confirmation.py --run-id long_111_seed142 --output-dir retraining/long111_seed142 --check-only
python tasks/2026-11-06_long111_full50k_v2/code/train_confirmation.py --run-id long_111_seed142 --output-dir retraining/long111_seed142
```

Use any of the30 IDs in `evidence/research_runs.json`. This research path uses the original40k/5k split and picks the best raw-validation epoch (earliest tie); unlike the final50k path it evaluates each epoch. The root `predict.py` accepts its exported `model.pt` for raw/flip inference. The unchanged original research modules are under `experiments/research_sweep/confirmation/`; historical full-sweep preparation is not needed for the single-run wrapper.

## Evaluation and review

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace tasks/2026-11-06_long111_full50k_v2/review.ipynb
python tasks/2026-11-06_long111_full50k_v2/code/evaluate_full50k.py --checkpoint tasks/2026-11-06_long111_full50k_v2/model.pt --labels independent_labels.csv --images independent_images --policy tta
```

The evaluation command is for independently supplied labeled data only; none is available for the final50k model in this release. It rejects the standard training CSV name, but the caller must establish genuine data independence. It prints accuracy and parameter count without inventing missing labels. The executed review notebook explicitly prints the final Kaggle accuracy as unavailable, and reports the earlier user-supplied 0.85 separately. Do not rename training labels and present training accuracy as test accuracy.

Validation during packaging: all 50k original images and normalization, all 240 sampler schedules, parameter counts, all 30 research count summaries, all 4 CSV schemas/IDs/ranges/hashes, independent CPU prediction replay, executed notebook, and PDF rendering. The standalone adapter also reproduced the first full50k epoch exactly (loss, accuracy, learning rate, sampling hash and update count). Training was not repeated for another 240 epochs during packaging; the original completed run and its full logs are included. Source/config hashes identify lineage; frozen selection evidence is preserved alongside a later metadata-only statistics report correction.

## Report and references

`report.pdf` is the course report (at most8 pages), with this public repository on page1. `review.ipynb` is executed and includes research comparisons and full50k training plots. `SHA256SUMS.json` covers release files. This release is additive and leaves v1 unchanged.

1. He et al. Deep Residual Learning for Image Recognition. CVPR2016. https://arxiv.org/abs/1512.03385
2. Zhang. Making Convolutional Networks Shift-Invariant Again. ICML2019. https://arxiv.org/abs/1904.11486
3. DeVries and Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. https://arxiv.org/abs/1708.04552
4. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR2014. https://jmlr.org/papers/v15/srivastava14a.html
5. Huang et al. Deep Networks with Stochastic Depth. ECCV2016. https://arxiv.org/abs/1603.09382
6. Loshchilov and Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR2017. https://arxiv.org/abs/1608.03983 (cosine schedule here, no restarts).
7. Course-provided Deep Learning Project1, Fall2026; https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules

Implementation is local project code inspired by these methods. No Kaggle submission, email or Brightspace upload is performed by these reproduction commands.
