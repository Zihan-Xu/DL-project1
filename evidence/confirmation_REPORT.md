# Confirmation study: seeds 142-144

This English edition describes the completed confirmation study. All result tables are regenerated from the accompanying CSV files. The [original report](https://github.com/Zihan-Xu/DL-project1/blob/c546dc1ab8e586bbffba663c820d3390290e9346/evidence/confirmation_REPORT.md) remains available in the earlier commit; translation provenance is recorded in [documentation_revision.json](documentation_revision.json).

## Design and completion

The study completed all 30 planned training runs, 3,360 epochs and both inference policies for each saved checkpoint. Every configuration used 40,000 training images and the same 5,000 validation images. Seeds were 142, 143 and 144. The core recipe was the 3x ResNet-20 with BlurPool, Cutout 12 and weight decay 0.0005.

In configuration names, R, D and S indicate rotation within +/-10 degrees, dropout 0.1 and maximum residual drop 0.1. The eight confirm_RDS configurations used 80 epochs. The two long_RDS configurations used 240 epochs. A 1 enables a factor; a 0 disables it. All models have 2,436,346 trainable parameters.

Checkpoints were selected by raw validation accuracy, with earlier epochs winning ties. Horizontal-flip test-time augmentation (TTA) then averaged the softmax probabilities from the original and mirrored image at that same checkpoint. The independent replay reproduced FP32 original and flipped logits for all 30 checkpoints. The 150,000 raw and 150,000 TTA prediction rows reuse the same 5,000 images; they are not 300,000 independent samples.

## Selection and findings

The specified selection rule chose **long_111 with TTA**, the highest mean among the 20 configuration/policy candidates. Validation accuracy was **96.0400%**, with seed SD **0.1510 percentage points**. Seeds 142, 143 and 144 obtained 96.06%, 95.88% and 96.18%: 4,803, 4,794 and 4,809 correct predictions out of 5,000. The full-data fit subsequently used the fixed seed 42. No confirmation seed was chosen for having the highest score.

At 80 epochs, the complete combination improved raw mean accuracy from 94.8267% to 95.4400%, a paired change of +0.6133 points. Extending the complete combination to 240 epochs added +0.5267 points. At 240 epochs, the complete combination exceeded the core by +0.5400 points. Their adjusted p-values are in the table below; none met the 0.05 threshold.

For long_111, TTA added +0.0733 points over raw inference. Its paired 95% interval was [-0.3962, 0.5429], with two seeds improving and one declining. TTA doubles forward passes and MACs without adding parameters. Raw inference averaged 95.9667% with seed SD 0.0503 points. TTA improved mean negative log likelihood (NLL) from 0.13885 to 0.12230, but mean worst-class recall declined from 91.733% to 91.333%. It did not improve every metric.

The 240-epoch complete combination took a mean 1,688.3 seconds for training and per-epoch validation, approximately 3.04 times the 80-epoch complete combination. Recorded training and validation for the entire study totaled 6.2502 GPU hours. Queue waiting, independent acceptance checks and the final full-data fit are excluded. No formal training failures were recorded.

| Configuration | Completed / planned seeds | Raw mean (%) | Seed SD (pp) | Training + validation (s) |
| --- | --- | --- | --- | --- |
| confirm_000 | 3 / 3 | 94.8267 | 0.1553 | 513.2 |
| confirm_001 | 3 / 3 | 95.0467 | 0.2139 | 530.2 |
| confirm_010 | 3 / 3 | 95.0800 | 0.1400 | 521.7 |
| confirm_011 | 3 / 3 | 94.9267 | 0.1137 | 543.7 |
| confirm_100 | 3 / 3 | 95.1733 | 0.1206 | 521.2 |
| confirm_101 | 3 / 3 | 95.3133 | 0.3107 | 543.4 |
| confirm_110 | 3 / 3 | 95.2333 | 0.2212 | 534.4 |
| confirm_111 | 3 / 3 | 95.4400 | 0.0600 | 554.8 |
| long_000 | 3 / 3 | 95.4267 | 0.3443 | 1549.4 |
| long_111 | 3 / 3 | 95.9667 | 0.0503 | 1688.3 |

## Both inference policies

MACs count convolution, linear layers and fixed blur filtering, excluding batch normalization, activations, additions and pooling. Raw inference uses 436,103,040 MACs per image; TTA uses 872,206,080.

| Configuration | Policy | Seeds | Mean (%) | SD (pp) | Macro F1 | Worst-class recall | NLL | MACs / image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_000 | raw | 3 | 94.8267 | 0.1553 | 0.94825 | 0.88600 | 0.17383 | 436,103,040 |
| confirm_000 | tta | 3 | 95.2333 | 0.1332 | 0.95233 | 0.89800 | 0.15154 | 872,206,080 |
| confirm_001 | raw | 3 | 95.0467 | 0.2139 | 0.95048 | 0.89333 | 0.16784 | 436,103,040 |
| confirm_001 | tta | 3 | 95.3867 | 0.2053 | 0.95388 | 0.89733 | 0.14356 | 872,206,080 |
| confirm_010 | raw | 3 | 95.0800 | 0.1400 | 0.95077 | 0.89067 | 0.16800 | 436,103,040 |
| confirm_010 | tta | 3 | 95.2400 | 0.1058 | 0.95240 | 0.89800 | 0.14724 | 872,206,080 |
| confirm_011 | raw | 3 | 94.9267 | 0.1137 | 0.94926 | 0.89200 | 0.16586 | 436,103,040 |
| confirm_011 | tta | 3 | 95.2667 | 0.1943 | 0.95268 | 0.90200 | 0.14572 | 872,206,080 |
| confirm_100 | raw | 3 | 95.1733 | 0.1206 | 0.95169 | 0.89867 | 0.16084 | 436,103,040 |
| confirm_100 | tta | 3 | 95.5267 | 0.1286 | 0.95525 | 0.90133 | 0.14003 | 872,206,080 |
| confirm_101 | raw | 3 | 95.3133 | 0.3107 | 0.95309 | 0.89600 | 0.15446 | 436,103,040 |
| confirm_101 | tta | 3 | 95.5333 | 0.2603 | 0.95529 | 0.90200 | 0.13783 | 872,206,080 |
| confirm_110 | raw | 3 | 95.2333 | 0.2212 | 0.95231 | 0.89733 | 0.15561 | 436,103,040 |
| confirm_110 | tta | 3 | 95.4667 | 0.3717 | 0.95461 | 0.89933 | 0.14036 | 872,206,080 |
| confirm_111 | raw | 3 | 95.4400 | 0.0600 | 0.95436 | 0.89467 | 0.15296 | 436,103,040 |
| confirm_111 | tta | 3 | 95.6600 | 0.0000 | 0.95657 | 0.90200 | 0.13793 | 872,206,080 |
| long_000 | raw | 3 | 95.4267 | 0.3443 | 0.95422 | 0.89600 | 0.15083 | 436,103,040 |
| long_000 | tta | 3 | 95.7333 | 0.2082 | 0.95733 | 0.90600 | 0.13219 | 872,206,080 |
| long_111 | raw | 3 | 95.9667 | 0.0503 | 0.95966 | 0.91733 | 0.13885 | 436,103,040 |
| long_111 | tta | 3 | 96.0400 | 0.1510 | 0.96039 | 0.91333 | 0.12230 | 872,206,080 |

## All 34 planned comparisons

Changes are measured in percentage points over common seeds. The 95% paired t intervals are unadjusted. Holm p-values correct the 24 training and 10 TTA comparisons together. All seeds were completed before this correction. No comparison passed Holm correction at 0.05, and all four factorial interaction intervals included zero.

| Comparison | Seeds | Mean change (pp) | 95% paired t interval | Holm p |
| --- | --- | --- | --- | --- |
| confirm_001__vs__confirm_000 | 3 | +0.2200 | [-0.1383, +0.5783] | 1 |
| confirm_010__vs__confirm_000 | 3 | +0.2533 | [-0.3599, +0.8665] | 1 |
| confirm_011__vs__confirm_000 | 3 | +0.1000 | [-0.3791, +0.5791] | 1 |
| confirm_100__vs__confirm_000 | 3 | +0.3467 | [-0.3279, +1.0212] | 1 |
| confirm_101__vs__confirm_000 | 3 | +0.4867 | [+0.1008, +0.8726] | 0.87283 |
| confirm_110__vs__confirm_000 | 3 | +0.4067 | [-0.5259, +1.3392] | 1 |
| confirm_111__vs__confirm_000 | 3 | +0.6133 | [+0.3139, +0.9128] | 0.39167 |
| long_000__vs__confirm_000 | 3 | +0.6000 | [-0.6411, +1.8411] | 1 |
| long_111__vs__confirm_111 | 3 | +0.5267 | [+0.2816, +0.7717] | 0.36786 |
| R_edge_confirm_101_confirm_001 | 3 | +0.2667 | [-0.2552, +0.7885] | 1 |
| R_edge_confirm_110_confirm_010 | 3 | +0.1533 | [-0.4204, +0.7270] | 1 |
| R_edge_confirm_111_confirm_011 | 3 | +0.5133 | [+0.3065, +0.7202] | 0.29433 |
| D_edge_confirm_011_confirm_001 | 3 | -0.1200 | [-0.8718, +0.6318] | 1 |
| D_edge_confirm_110_confirm_100 | 3 | +0.0600 | [-0.2029, +0.3229] | 1 |
| D_edge_confirm_111_confirm_101 | 3 | +0.1267 | [-0.5479, +0.8012] | 1 |
| S_edge_confirm_011_confirm_010 | 3 | -0.1533 | [-0.7705, +0.4639] | 1 |
| S_edge_confirm_101_confirm_100 | 3 | +0.1400 | [-0.9174, +1.1974] | 1 |
| S_edge_confirm_111_confirm_110 | 3 | +0.2067 | [-0.4418, +0.8551] | 1 |
| interaction_RD | 3 | +0.0267 | [-0.0907, +0.1441] | 1 |
| interaction_RS | 3 | +0.1400 | [-0.6561, +0.9361] | 1 |
| interaction_DS | 3 | -0.1533 | [-0.7123, +0.4056] | 1 |
| interaction_RDS | 3 | +0.4400 | [-0.5524, +1.4324] | 1 |
| long_111__vs__long_000 | 3 | +0.5400 | [-0.3390, +1.4190] | 1 |
| interaction_stack_length | 3 | -0.0733 | [-1.2452, +1.0985] | 1 |
| confirm_000__tta_vs_raw | 3 | +0.4067 | [+0.1031, +0.7102] | 0.80655 |
| confirm_001__tta_vs_raw | 3 | +0.3400 | [+0.1609, +0.5191] | 0.43996 |
| confirm_010__tta_vs_raw | 3 | +0.1600 | [-0.1422, +0.4622] | 1 |
| confirm_011__tta_vs_raw | 3 | +0.3400 | [-0.0931, +0.7731] | 1 |
| confirm_100__tta_vs_raw | 3 | +0.3533 | [+0.2099, +0.4968] | 0.29433 |
| confirm_101__tta_vs_raw | 3 | +0.2200 | [+0.0213, +0.4187] | 1 |
| confirm_110__tta_vs_raw | 3 | +0.2333 | [-0.2007, +0.6674] | 1 |
| confirm_111__tta_vs_raw | 3 | +0.2200 | [+0.0710, +0.3690] | 0.69333 |
| long_000__tta_vs_raw | 3 | +0.3067 | [-0.1550, +0.7683] | 1 |
| long_111__tta_vs_raw | 3 | +0.0733 | [-0.3962, +0.5429] | 1 |

## Limits and follow-up decision

The study used validation results for configuration selection and did not use the historical local-test or competition images for that decision. New training seeds do not create a new test dataset. Reusing one validation split, selecting an epoch and screening candidates can make the selected result optimistic. An interval spanning zero is inconclusive, not evidence of equivalence.

After the planned comparisons finished, the decision was to fit the selected recipe once on all permitted training images with seed 42. No extra seeds were added in pursuit of statistical significance, and no additional long-training factorial grid was run for this selection. This was a ranking among measured candidates, not proof of a globally optimal model.

Derived comparisons were recalculated by combining integer correct counts before converting them to percentage points. This numerical revision changed neither the selected candidate nor any of the 34 Holm decisions; the largest absolute change in an adjusted p-value was approximately 9.1e-14. It did not alter training or predictions.

The numeric records are [confirmation_summary.csv](confirmation_summary.csv), [confirmation_inference_summary.csv](confirmation_inference_summary.csv), [confirmation_comparisons.csv](confirmation_comparisons.csv) and [confirmation_inference_run_metrics.csv](confirmation_inference_run_metrics.csv). Per-image predictions and logs are in [research_logs/](research_logs/). The original training workflow is retained in the [archived v2 release](https://github.com/Zihan-Xu/DL-project1/tree/main/tasks/2026-11-06_long111_full50k_v2).
