"""Write English summaries of new runs without changing their training or selection."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED = {
    "architecture": 42,
    "data": 71,
    "training": 120,
    "combinations": 86,
    "confirmation": 30,
}


def collect(folder, category):
    pattern = (
        "*/seed_*/metrics.json" if category == "training" else "runs/*/metrics.json"
    )
    records = []
    validation_identity = None
    for path in sorted(folder.glob(pattern)):
        metric = json.loads(path.read_text(encoding="utf-8-sig"))
        if metric.get("complete", True) is not True:
            continue
        key = (
            "config_id"
            if category == "training"
            else "arm" if category == "data" else "configuration"
        )
        with (path.parent / "validation_predictions.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            predictions = list(csv.DictReader(file))
        identity = [(r["id"], r.get("label", r.get("true_label"))) for r in predictions]
        assert (
            len(identity)
            == len({r[0] for r in identity})
            == int(metric["validation_n"])
            == 5000
        )
        if validation_identity is None:
            validation_identity = identity
        assert (
            identity == validation_identity
        ), "All comparisons must use the same ordered validation set"
        correct = sum(
            label == row.get("prediction", row.get("predicted_label"))
            for (_, label), row in zip(identity, predictions)
        )
        assert correct == int(metric["validation_correct"]), str(path)
        records.append(
            {
                "configuration": metric[key],
                "seed": int(metric["seed"]),
                "correct": correct,
                "n": 5000,
                "accuracy_percent": correct / 50,
            }
        )
    assert len({(r["configuration"], r["seed"]) for r in records}) == len(records)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["configuration"]].append(record["accuracy_percent"])
    summary = [
        {
            "configuration": key,
            "seeds": len(values),
            "mean_accuracy_percent": statistics.mean(values),
            "sd_accuracy_pp": statistics.stdev(values) if len(values) > 1 else None,
        }
        for key, values in sorted(grouped.items())
    ]
    return records, summary


def summarize(folder, category, final=False):
    folder = Path(folder)
    records, summary = collect(folder, category)
    expected = EXPECTED[category]
    if final:
        assert (
            len(records) == expected
        ), f"Only {len(records)} of {expected} planned outputs are present"
    for name, values in [
        ("reproduction_runs.csv", records),
        ("reproduction_summary.csv", summary),
    ]:
        if values:
            with (folder / name).open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    lines = [
        f"# {category.capitalize()} reproduction results",
        "",
        f"Verified completed metric files: {len(records)} / {expected}.",
        "",
        "Accuracy is recomputed from each saved validation prediction table. These results use the same validation images across seeds; they are not competition scores. The table reports means and sample standard deviations. It does not perform the original full acceptance audit or multiple-comparison analysis.",
        "",
        "| Configuration | Completed seeds | Mean accuracy (%) | Seed SD (pp) |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary:
        sd = (
            "pending"
            if row["sd_accuracy_pp"] is None
            else f'{row["sd_accuracy_pp"]:.4f}'
        )
        lines.append(
            f'| {row["configuration"]} | {row["seeds"]} | {row["mean_accuracy_percent"]:.4f} | {sd} |'
        )
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
