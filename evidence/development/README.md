# Development evidence

Numeric records supporting the expanded course report. Sources and current SHA-256 hashes are listed in `sources.json`; frozen originals remain in the project archive.

The width and initial augmentation studies used one seed. Later studies used their own matched controls; different stages are not one controlled comparison. Architecture: 42 runs. Data: 71 results, including six verified historical imports (65 new runs). Training: 120 outputs from 107 physical training trajectories, because averaging methods share trajectories. Combinations: 86 runs. The separate final confirmation study contains 30 runs. All figures are local validation results, not Kaggle scores.

The report builder checks completion, recomputes summary statistics from per-run records, and reads its tables from these CSV files. No training images or correspondence are included.

The interpretation column of `data_comparisons.csv` is translated into English. Its numeric fields are unchanged. `sources.json` retains the original hash and file link alongside the current hash. Original acceptance JSON files describe the source artifacts and retain their historical hashes.
