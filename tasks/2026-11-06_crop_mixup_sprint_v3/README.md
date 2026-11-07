# Crop and Mixup: completed six-run deadline study

Author and Kaggle team: **Zihan Xu**. November 6, 2026.

All six registered runs completed 80 epochs and passed independent CPU prediction replay. The best new configuration/policy by the complete three-seed mean is **crop_mixup/tta: 95.7800%** on the fixed 5,000-image validation set. It does not exceed the previous 240-epoch recipe's 96.04% research mean. Those recipes have different training budgets; this is not a controlled epoch comparison. The paired 80-epoch control is reported separately.

The separately registered fixed dual-residual experiment achieved **96.3333%** over all three matched seeds, with **4,872,692 parameters counting both branches**. Its recommendation criterion passed: **True**. This is a secondary experiment on the same validation set, not a fresh holdout. All three CPU replays passed; weights remain fixed at 0.5/0.5, with no mixture or temperature search.

**Recommended overall:** `submissions/00_RECOMMENDED_dual_residual_full50k/submission.csv`.
**Selected single-branch candidate:** `submissions/01_SELECTED_crop_mixup_full50k_tta/submission.csv`.
New candidate Kaggle accuracy is **unavailable** until an actual submission is scored. The previous long_111/full50k/TTA CSV has user-reported Kaggle score **0.87900**; remote uploaded bytes were not independently retrieved. No Kaggle submission is performed here.

## Experiment and architecture

The control is the accepted confirm_000 runs at seeds 142,143,144. Two added recipes use the same three seeds, split, 80-epoch cosine schedule, sampler, batch 128, SGD lr 0.1/momentum 0.9/weight decay 0.0005. Crop adds reflection padding 4 and a uniform32x32 crop. Crop+Mixup additionally mixes each batch with probability 0.5 and Beta(0.2,0.2) image/label weights. Both retain horizontal flip p0.5 and Cutout 12 p0.5. Rotation, dropout, stochastic depth, label smoothing, attention and teacher distillation are absent in this round.

The residual network is a width3 ResNet20: channels 48/96/192, three blocks per stage, convolution-BN-ReLU residual paths, skip projections when shape changes, BlurPool downsampling, global average pooling, and a 10-class linear head. It has **2,436,346 trainable parameters**, below 5 million. All weights are trained from scratch using only course training images.

Research training uses40,000 images and the same stratified5,000-image validation set as previous experiments. Each run completes all 80 epochs; its highest raw-validation checkpoint is retained, with earliest-epoch tie breaking. Raw and equal-weight original/horizontal-mirror softmax averaging are evaluated at that fixed checkpoint. The complete three-seed mean selects configuration/policy; deployment seed42 is fixed in advance. The old local-test split is not treated as pristine after earlier use.

The final candidate is a fresh80-epoch seed42 fit on all 50,000 labeled images, using the last prescribed epoch and training-only normalization recomputed on all 50k. It processes4,000,000 image presentations and31,280 optimizer updates. Previous holdouts are now training data: **there is no independent local accuracy for this final model**. Mixed-batch training accuracy is not an interpretable comparison metric.

## Reproduce prediction

From the repository root, supply authorized data in `data/competition/`: original `train_images`, `test_images`, `train_labels.csv`, `test.csv`, and `sample_submission.csv`. No course image files are published. Install the repository requirements with the recorded PyTorch2.11.0+cu128 environment for closest numerical agreement. Training used Python3.13.5, NumPy2.1.3 and RTX5070Ti; full details are in `evidence/environment.json`.

Install this task's requirements with `python -m pip install -r tasks/2026-11-06_crop_mixup_sprint_v3/requirements.txt`; for the recorded CUDA build, install `torch==2.11.0` from `https://download.pytorch.org/whl/cu128`. All installed package versions are recorded in `evidence/package_versions.json`.

```powershell
python tasks/2026-11-06_crop_mixup_sprint_v3/code/predict.py --data-dir data/competition --policy tta --output sprint_reproduced.csv --compare tasks/2026-11-06_crop_mixup_sprint_v3/submissions/01_SELECTED_crop_mixup_full50k_tta/submission.csv
```

Use `--policy raw` or `--policy tta` for the two single-model inference choices. The checkpoint defaults to this task's `model.pt`. Output files must not already exist. These two policies use one model with one or two FP32 passes.

For the separate dual model, run `python tasks/2026-11-06_crop_mixup_sprint_v3/code/predict_dual.py --data-dir data/competition --output dual.csv`. Its default `ensemble.pt` contains both full50k checkpoints: new80 epochs and reference240 epochs, both seed42, both trained from scratch on course data. Their probabilities are averaged equally after each branch's selected inference policy. Training costs are80+240 epochs, not80 epochs; parameter count includes every parameter in both branches. Reproduce the reference using the previous `2026-11-06_long111_full50k_v2` release's training instructions, then run `code/assemble_dual.py --new NEW.pt --reference REFERENCE.pt --policy tta --output ensemble.pt`. No extra fitting occurs during assembly.

The competition architecture rule allows custom residual CNN components and limits all trainable parameters to5M; no explicit ensemble ban was present in the reviewed rules. Counting both complete branches gives4,872,692. This is our reading of the rules, not special staff approval. New weights and predictions are published only after the competition deadline to avoid sharing between teams during competition.

Evaluate the complete dual model using `python tasks/2026-11-06_crop_mixup_sprint_v3/code/evaluate_dual.py --labels independent_labels.csv --images independent_images`. The label CSV must contain `filename,label`; the caller must establish that these images were not used for training or selection. No such independent new labeled set was available for the final50k models.

## Reproduce training and evaluation

```powershell
python tasks/2026-11-06_crop_mixup_sprint_v3/code/train_full50k.py --data-dir data/competition --output-dir retraining/sprint --check-only
python tasks/2026-11-06_crop_mixup_sprint_v3/code/train_full50k.py --data-dir data/competition --output-dir retraining/sprint
python tasks/2026-11-06_crop_mixup_sprint_v3/code/evaluate_full50k.py --checkpoint tasks/2026-11-06_crop_mixup_sprint_v3/model.pt --labels independent_labels.csv --images independent_images --policy tta
python -m jupyter nbconvert --execute --to notebook --inplace tasks/2026-11-06_crop_mixup_sprint_v3/review.ipynb
```

The standalone training adapter checks original image bytes, normalization, all 80 sampler schedules, and parameter count before optimization. It reuses the executed full-data engine with an output-directory lock. Use a free CUDA/BF16 GPU; this portable lock does not coordinate other GPU jobs. Rerun the same command to resume saved optimizer, scheduler and RNG states. Store outputs outside this immutable release. Independent evaluation requires genuinely new labeled data; renaming training labels does not make them independent.

`code/base_operators.py` and `architecture_model.py` preserve the original model/sampling/schedule implementations. The portable crop/Mixup module changes only its import and adds aliases. Exact pixel, initialization, loss and gradient parity is recorded in `evidence/portable_operator_checks.json`; the training main/save/CSV ASTs match the executed engine. The original research entry points and audit are archived in `source_archive`, and six selected research checkpoints plus complete epoch logs/predictions are included under `evidence/research_logs`.

For a research rerun, first prepare the original cache with the repository-root `prepare_data.py`, then run `python tasks/2026-11-06_crop_mixup_sprint_v3/code/train_research.py --run-id crop_seed142 --output-dir retraining/crop142 --check-only`. Remove `--check-only` to train; any of the six IDs in the manifest is supported. This uses the original40k/5k split and actual training engine. The standalone full50k entry point reproduces the delivered final candidate without requiring the historical experiment queue. The notebook recomputes every new run's counts from probabilities and all six means; original execution evidence is preserved rather than claiming another complete six-run retraining during packaging.

## Interpretation and limits

Three seeds are repeats on the same data split, not three independent datasets. Paired t intervals describe seed variation; six exploratory comparisons use Holm correction. Repeated prior model selection on the same 5k set limits generalization claims. Better validation accuracy does not guarantee a better Kaggle score. CutMix, AugMix, RandAugment and a new WideResNet were researched but **not run** in this finite round. Their original sources are in the experiment sheet.

The delivered CSVs have exactly 2,000 unique IDs in sample order, `id,label` columns and integer labels 0-9. Per-version metadata records model/config/policy and SHA-256; disagreement counts show that the alternatives are distinct. `report.pdf` is at most8pages; `review.ipynb` is executed. The spreadsheet separates Kaggle scores, research validation and pending results. No credentials, raw images, messages or caches are included.

## References

1. He et al., Deep Residual Learning for Image Recognition, CVPR2016. https://arxiv.org/abs/1512.03385
2. Zhang, Making Convolutional Networks Shift-Invariant Again, ICML2019. https://arxiv.org/abs/1904.11486
3. DeVries and Taylor, Improved Regularization of Convolutional Neural Networks with Cutout, 2017. https://arxiv.org/abs/1708.04552
4. Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR2018. https://arxiv.org/abs/1710.09412
5. Yun et al., CutMix, ICCV2019. https://arxiv.org/abs/1905.04899
6. Hendrycks et al., AugMix, ICLR2020. https://arxiv.org/abs/1912.02781
7. Cubuk et al., RandAugment, 2019. https://arxiv.org/abs/1909.13719
8. Zagoruyko and Komodakis, Wide Residual Networks, BMVC2016. https://arxiv.org/abs/1605.07146
9. Course Project1, Fall2026; https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules
10. Loshchilov and Hutter, SGDR, ICLR2017. https://arxiv.org/abs/1608.03983 (cosine schedule here, without restarts).
11. Srivastava et al., Dropout, JMLR2014. https://jmlr.org/papers/v15/srivastava14a.html
12. Huang et al., Deep Networks with Stochastic Depth, ECCV2016. https://arxiv.org/abs/1603.09382
13. Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NeurIPS2017. https://arxiv.org/abs/1612.01474
