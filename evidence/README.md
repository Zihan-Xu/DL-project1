# Evidence and score provenance

The numeric research and full-data training records in this directory come from the original second-version release. Their original paths and pre-submission fields are historical provenance. In particular, `RESULT.json` records the training-time state when the Kaggle score was unknown; it is not the current score record.

`kaggle_submission_verified.json` is the later verification record: the CSV downloaded from the user's second Kaggle submission matched the recommended local file byte for byte, with SHA-256 `f25fd0618bf2784593ca29a875839eb6e08160a58db66adf7f9c02afc5c31996`. It records public accuracy 0.87900 and submission time 2026-11-06 22:15:04 America/New_York. Private accuracy is unknown. The current two CSV variants are described by `../submissions/versions.json`.

`clean_code_equivalence.json`, `clean_epoch1_replay.json`, `clean_prediction_tta.json`, `clean_prediction_raw.json` and `clean_acceptance.json` document the current cleanup checks. They do not represent a new model-selection experiment or a repeated 240-epoch final fit.

All 30 confirmation runs and both inference policies are represented in `research_logs/`. Their original training workflow is supplied in `../research/`, alongside the earlier development studies. The public `main` branch preserves the historical releases. Recompute their saved prediction counts using `verify.py` or the executed notebook in this branch.

No private messages, raw course images, credentials or training caches are included.

`development/` adds numeric records from the earlier baseline, augmentation, sampling, architecture, training and combination studies for the expanded report. It contains source hashes and original acceptance records; these are historical experiments, not new training performed for this report revision.

## English editions

`confirmation_REPORT.md` is the English edition of the earlier confirmation narrative. Its 10 configuration rows, 20 inference-policy rows and 34 paired comparisons are regenerated from the unchanged CSV records. In `development/data_comparisons.csv`, only the interpretation column was translated. All other fields, including every numeric value, are unchanged.

[documentation_revision.json](documentation_revision.json) records the original and current hashes and links to the original files. `development/sources.json` identifies the translated comparison table explicitly. Historical acceptance records retain the hashes of the original artifacts they audited; the current release manifest covers the English editions. The frozen release directories and earlier commits remain available for source verification.

`final_delivery_review.json` records the final requirement and consistency review. `final_prediction_replay.json` is a fresh CPU replay on all 2,000 original competition PNGs; every label matched the second submitted CSV. This review did not retrain the full model or create a new competition submission.
