"""Run the predeclared width sweep and select by validation Top-1 before local testing."""
import argparse
import json

import torch
from torch import nn

from model import ResNet20
from prepare_data import OUT, ROOT, sha256, write_json
from run_baseline import arrays, evaluate, normalizer, predict, rows, train, write_rows


STUDY = ROOT / "experiments" / "width_sweep"
WIDTHS = (1, 2, 3, 4)


def directory(width):
    return OUT if width == 1 else STUDY / f"{width}x"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def forward_macs(width):
    """Count convolution and linear multiply-accumulates per 32x32 RGB image."""
    total = 0

    def count(layer, inputs, output):
        nonlocal total
        total += output.numel() * layer.weight[0].numel()

    with torch.device("meta"):
        net = ResNet20(width).eval()
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_forward_hook(count)
        net(torch.empty(1, 3, 32, 32))
    return total


def compare(device, audit, manifest, baseline_config):
    images, labels, val_rows = arrays("validation", manifest, device)
    norm = normalizer(audit["train_rgb_mean"], audit["train_rgb_std"], device)
    summaries = []
    validation_predictions = []
    for width in WIDTHS:
        folder = directory(width)
        run = read_json(folder / "run.json")
        cfg = run["config"]
        expected_cfg = dict(baseline_config, width_multiplier=width) if width > 1 else baseline_config
        if cfg != expected_cfg or run["split_sha256"] != audit["split_sha256"]:
            raise ValueError(f"{width}x recipe or split differs from plan")
        history = rows(folder / "history.csv")
        if [int(row["epoch"]) for row in history] != list(range(1, cfg["epochs"] + 1)):
            raise ValueError(f"Incomplete epoch history for {width}x")
        best = max(history, key=lambda row: float(row["validation_accuracy"]))
        ckpt = torch.load(folder / "resnet20.pt", map_location="cpu", weights_only=True)
        if ckpt["config"] != cfg or ckpt["split_sha256"] != audit["split_sha256"] or ckpt["epoch"] != int(best["epoch"]):
            raise ValueError(f"{width}x checkpoint does not match its training record")
        model = ResNet20(width).to(device=device, memory_format=torch.channels_last)
        model.load_state_dict(ckpt["model_state"])
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params != run["trainable_parameters"] or params > 5_000_000:
            raise ValueError(f"{width}x parameter count mismatch")
        predicted = predict(model, images, norm, cfg["batch_size"]).argmax(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        if not (accuracy == float(best["validation_accuracy"]) == ckpt["validation_accuracy"]):
            raise ValueError(f"{width}x validation accuracy did not reproduce")
        validation_predictions.append(predicted.tolist())
        summaries.append({
            "width": width, "channels": "/".join(str(width*c) for c in (16, 32, 64)),
            "parameters": params, "macs_per_image": forward_macs(width),
            "best_epoch": int(best["epoch"]), "validation_correct": correct,
            "validation_n": len(labels), "validation_accuracy": accuracy, "best_epoch_val_loss": float(best["validation_loss"]),
            "last_epoch_val_accuracy": float(history[-1]["validation_accuracy"]),
            "training_seconds": run["training_seconds"], "checkpoint_sha256": sha256(folder / "resnet20.pt"),
        })
    for result in summaries:
        result["delta_vs_1x_pp"] = round((result["validation_accuracy"] - summaries[0]["validation_accuracy"]) * 100, 2)
    write_rows(STUDY / "comparison.csv", list(summaries[0]), (r.values() for r in summaries))
    write_rows(STUDY / "validation_predictions.csv", ["id", "label"] + [f"pred_{w}x" for w in WIDTHS],
               zip([r["id"] for r in val_rows], labels.tolist(), *validation_predictions))
    # WIDTHS are ascending, so a tie in correct predictions resolves to fewer parameters.
    chosen = max(summaries, key=lambda r: r["validation_correct"])
    write_json(STUDY / "selection.json", {"rule": "max validation correct; tie -> smaller width",
               "split_sha256": audit["split_sha256"], "selected": chosen})
    print(f"Selected {chosen['width']}x on validation: {chosen['validation_accuracy']:.2%}", flush=True)
    if chosen["width"] != 1:
        # The original held-out evaluation ran in a fresh process with this default.
        torch.backends.cudnn.deterministic = False
        evaluate(device, audit, manifest, out=directory(chosen["width"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "compare"))
    parser.add_argument("--widths", type=int, nargs="+", choices=WIDTHS[1:], default=WIDTHS[1:])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    audit = read_json(OUT / "data_audit.json")
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Split manifest was changed after auditing")
    cfg = read_json(OUT / "run.json")["config"]
    if cfg != read_json(ROOT / "config.json"):
        raise ValueError("Current recipe differs from recorded baseline recipe")
    manifest = rows(OUT / "splits.csv")
    device = torch.device(args.device)
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.action == "compare":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        compare(device, audit, manifest, cfg)
        return
    for width in args.widths:
        folder = directory(width)
        folder.mkdir(exist_ok=False)
        print(f"Training {width}x CIFAR ResNet-20", flush=True)
        train(dict(cfg, width_multiplier=width), device, audit, manifest, out=folder)


if __name__ == "__main__":
    main()
