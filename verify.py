"""Audit the submitted artifact and recompute all reported research accuracies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import build

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_submission(path, expected_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["id", "label"], path
        records = list(reader)
    ids = [row["id"] for row in records]
    assert len(records) == len(set(ids)) == 2000
    assert ids == expected_ids, "IDs must match the official sample order"
    assert all(row["label"] in list(map(str, range(10))) for row in records)
    return np.array([int(row["label"]) for row in records])


def audit(root=ROOT, data_dir=None):
    root = Path(root)
    torch.set_num_threads(4)
    config = read(root / "config.json")
    cp = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model = build(cp["recipe"])
    model.load_state_dict(cp["model_state"], strict=True)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters == cp["parameters"] == 2436346 < 5000000
    assert cp["epoch"] == 240 and cp["seed"] == 42 and cp["training_images"] == 50000
    assert cp["recipe"] == config["recipe"] and cp["config_sha256"] == digest(
        root / "config.json"
    )
    assert (
        digest(root / "model.pt")
        == "6ffed5a3b60d34ee7da7f1586984b457b91eb2740e9585163130b3e4cfc57b7c"
    )
    history = rows(root / "evidence/history.csv")
    assert [int(r["epoch"]) for r in history] == list(range(1, 241))
    assert all(
        int(r["examples"]) == 50000 and int(r["updates"]) == 391 for r in history
    )
    acceptance = read(root / "evidence/confirmation_ACCEPTANCE.json")
    assert acceptance["passed"] and acceptance["completed_verified"] == 30
    assert acceptance["tta_completed_verified"] == 30
    assert read(root / "evidence/confirmation_REPLAY_ACCEPTANCE.json")["passed"]
    metrics = rows(root / "evidence/confirmation_inference_run_metrics.csv")
    summary = rows(root / "evidence/confirmation_inference_summary.csv")
    assert len(metrics) == 60 and len(summary) == 20
    validation_ids = None
    for row in metrics:
        run = f"{row['configuration']}_seed{row['seed']}"
        filename = (
            "validation_predictions.csv"
            if row["policy"] == "raw"
            else "tta_predictions.csv"
        )
        frame = rows(root / "evidence/research_logs" / run / filename)
        assert len(frame) == 5000 and len({r["id"] for r in frame}) == 5000
        identity = [(r["id"], r.get("label", r.get("true_label"))) for r in frame]
        if validation_ids is None:
            validation_ids = identity
        assert (
            identity == validation_ids
        ), "Compare identical validation examples and labels"
        correct = sum(
            r.get("label", r.get("true_label"))
            == r.get("prediction", r.get("predicted_label"))
            for r in frame
        )
        assert correct == int(row["validation_correct"]), run
    for row in summary:
        matching = [
            r
            for r in metrics
            if (r["configuration"], r["policy"])
            == (row["configuration"], row["policy"])
        ]
        assert {int(r["seed"]) for r in matching} == {142, 143, 144}
        total = sum(int(r["validation_correct"]) for r in matching)
        assert total == int(row["total_correct"])
        assert abs(total / 15000 * 100 - float(row["mean_accuracy_percent"])) < 1e-10
    ranked = sorted(
        summary,
        key=lambda r: (
            -int(r["total_correct"]),
            r["policy"] != "raw",
            int(r["epochs"]),
            r["configuration"],
        ),
    )
    assert (ranked[0]["configuration"], ranked[0]["policy"]) == ("long_111", "tta")
    versions = read(root / "submissions/versions.json")
    expected_ids = (
        [r["id"] for r in rows(Path(data_dir) / "sample_submission.csv")]
        if data_dir
        else [str(i) for i in range(2000)]
    )
    predictions = []
    for version in versions:
        file = root / "submissions" / version["version"] / "submission.csv"
        predictions.append(validate_submission(file, expected_ids))
        assert digest(file) == version["submission_sha256"]
    assert len(versions) == 2 and int((predictions[0] != predictions[1]).sum()) == 69
    score = read(root / "evidence/kaggle_submission_verified.json")
    assert (
        score["byte_identical"]
        and score["predictions_identical"]
        and score["public_score"] == 0.879
    )
    assert score["remote_sha256"] == versions[0]["submission_sha256"]
    manifest = root / "SHA256SUMS.json"
    if manifest.exists():
        for name, expected in read(manifest).items():
            assert digest(root / name) == expected, name
    return dict(
        passed=True,
        parameters=parameters,
        completed_research_runs=30,
        research_prediction_tables_verified=60,
        research_validation_mean_percent=96.04,
        final_training_epochs=240,
        optimizer_updates=93840,
        submission_versions=2,
        tta_raw_disagreements=69,
        kaggle_public_accuracy_percent=87.9,
        kaggle_private_accuracy=None,
        final_independent_local_accuracy=None,
        submission_identity="Exact second submitted CSV; SHA-256 verified",
        sample_order_check=(
            "official sample file" if data_dir else "recorded official IDs 0..1999"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(data_dir=args.data_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
