# Evidence and score provenance

The research and full-data training records in this directory are unchanged snapshots from the original second-version release. Their original paths and pre-submission fields are historical provenance. In particular, `RESULT.json` records the training-time state when the Kaggle score was unknown; it is not the current score record.

`kaggle_submission_verified.json` is the later verification record: the CSV downloaded from the user's second Kaggle submission matched the recommended local file byte for byte, with SHA-256 `f25fd0618bf2784593ca29a875839eb6e08160a58db66adf7f9c02afc5c31996`. It records public accuracy 0.87900 and submission time 2026-11-06 22:15:04 America/New_York. Private accuracy is unknown. The current two CSV variants are described by `../submissions/versions.json`.

`clean_code_equivalence.json`, `clean_epoch1_replay.json`, `clean_prediction_tta.json`, `clean_prediction_raw.json` and `clean_acceptance.json` document the current cleanup checks. They do not represent a new model-selection experiment or a repeated 240-epoch final fit.

All 30 confirmation runs and both inference policies are represented in `research_logs/`. Their full training reproduction workflow is preserved on the public `main` archive branch. Recompute their saved prediction counts using `verify.py` or the executed notebook in this branch.

No private messages, raw course images, credentials or training caches are included.
