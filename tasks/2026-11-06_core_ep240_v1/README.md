# core_ep240 v1 - 2026-11-06

Author and Kaggle team: **Zihan Xu**.

The highest observed recipe mean in the accepted 86-run combination study is 95.3867% raw validation accuracy (seeds42/43/44). The same checkpoints average 95.7667% with horizontal-flip probability averaging. Seed42 remains the fixed deployment seed.

| Version | Local validation | Kaggle score | Use |
|---|---:|---:|---|
| [seed42 flip](submissions/01_recommended_seed42_flip/submission.csv) | 95.78% | Unavailable | Recommended |
| [seed42 raw](submissions/02_reference_seed42_raw/submission.csv) | 95.26% | 85.00%, user-reported | Reference |
| [seed43 flip](submissions/03_alternative_seed43_flip/submission.csv) | 95.68% | Unavailable | Alternate training seed |
| [seed44 flip](submissions/04_alternative_seed44_flip/submission.csv) | 95.84% | Unavailable | Alternate training seed |

Each CSV uses one checkpoint with 2,436,346 parameters. No model ensemble is included. Different seeds are alternatives; selecting the highest single validation run after the fact would change the preselected deployment policy.

- [Four-page report](report.pdf): architecture, training, comparison results, limitations, citations and repository URL.
- [Executed review notebook](review.ipynb): parameter counts, validation calculations, reported test-score provenance, CSV integrity and plots.
- [Version metadata](versions.json): exact CSV hashes and inference policies.
- [Models](models/): portable weights for all three seeds.
- [Evidence](evidence/): source configurations, logs, comparison table, probabilities and acceptance records.
- [Root README](../../README.md): installation, data preparation, single-run training and CSV-reproduction commands.

No new Kaggle submission was performed during packaging. These files are an immutable release; later completed and independently accepted work may supersede it in a new directory.
