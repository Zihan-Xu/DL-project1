"""Run the fixed recipe or evaluate and predict from its saved checkpoint."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import ResNet20
from prepare_data import CACHE, DATA, OUT, ROOT, sha256, write_json


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


def normalizer(mean, std, device):
    return tuple(torch.tensor(v, dtype=torch.float32, device=device).view(1, 3, 1, 1)
                 for v in (mean, std))


def normalize(batch, mean, std):
    return (batch.permute(0, 3, 1, 2).float().div_(255) - mean) / std


def padded_images(images, pad):
    return F.pad(images.permute(0, 3, 1, 2), (pad,) * 4, mode="reflect").permute(0, 2, 3, 1).contiguous()


def crop_flip(padded, idx, pad, flip_probability):
    # Independent uniform crop offsets and independent left-right flips per image.
    device = padded.device
    offsets = torch.randint(2 * pad + 1, (2, len(idx), 1, 1), device=device)
    axis = torch.arange(32, device=device)
    flip = torch.rand(len(idx), 1, 1, device=device) < flip_probability
    cols = torch.where(flip, 31 - axis, axis) + offsets[1]
    return padded[idx[:, None, None], offsets[0] + axis[:, None], cols]


@torch.inference_mode()
def predict(model, images, norm, batch_size):
    model.eval()
    return torch.cat([model(normalize(part, *norm)) for part in images.split(batch_size)])


def arrays(split, manifest, device):
    indices = [i for i, row in enumerate(manifest) if row["split"] == split]
    raw = np.load(CACHE / "train.npy", mmap_mode="r")
    images = torch.from_numpy(raw[indices]).to(device)
    labels = torch.tensor([int(manifest[i]["label"]) for i in indices], device=device)
    return images, labels, [manifest[i] for i in indices]


def train(cfg, device, audit, manifest, *, out=OUT, augmentation=None, epoch_batches=None):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    images, y, _ = arrays("train", manifest, device)
    val_images, val_y, _ = arrays("validation", manifest, device)
    if augmentation is None:
        padded = padded_images(images, cfg["crop_padding"])
        del images
    norm = normalizer(audit["train_rgb_mean"], audit["train_rgb_std"], device)
    model = ResNet20(cfg.get("width_multiplier", 1)).to(device=device, memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params > 5_000_000:
        raise ValueError(f"Trainable parameters exceed course budget: {params}")
    # FP32 weights and optimizer state; CUDA BF16 autocast for training only.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    metadata = {
        "config": cfg, "trainable_parameters": params, "split_sha256": audit["split_sha256"],
        "mean": audit["train_rgb_mean"], "std": audit["train_rgb_std"],
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "training_autocast": "bfloat16" if use_bf16 else "off",
        "evaluation_autocast": "off", "cudnn_deterministic": True, "cudnn_benchmark": False,
        "platform": platform.platform(),
    }
    write_json(out / "run.json", metadata)
    print(f"Device={metadata['device']}, parameters={params:,}, train={len(y)}, validation={len(val_y)}", flush=True)
    path = out / "resnet20.pt"
    best = -1
    training_examples_seen = 0
    start = time.monotonic()
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "lr", "train_loss", "train_accuracy", "validation_loss",
                         "validation_accuracy", "epoch_seconds"])
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            model.train()
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            batches = (torch.randperm(len(y), device=device).split(cfg["batch_size"])
                       if epoch_batches is None else epoch_batches(epoch, device, cfg["batch_size"]))
            examples_seen = 0
            for idx in batches:
                if augmentation is None:
                    batch = crop_flip(padded, idx, cfg["crop_padding"], cfg["horizontal_flip_probability"])
                    x = normalize(batch, *norm)
                else:
                    x = augmentation(images, idx, norm)
                labels = y[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                totals[0] += loss.detach() * len(idx)
                totals[1] += (logits.detach().argmax(1) == labels).sum()
                examples_seen += len(idx)
            if examples_seen == 0:
                raise ValueError("The training sampler produced no examples")
            training_examples_seen += examples_seen
            val_logits = predict(model, val_images, norm, cfg["batch_size"])
            val_loss = F.cross_entropy(val_logits, val_y).item()
            correct = (val_logits.argmax(1) == val_y).sum().item()
            val_acc = correct / len(val_y)
            train_loss, train_acc = (totals / examples_seen).tolist()
            if not np.isfinite([train_loss, val_loss]).all():
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            if correct > best:
                best = correct
                tmp = path.with_suffix(".tmp")
                torch.save({**metadata, "epoch": epoch, "validation_accuracy": val_acc,
                            "model_state": model.state_dict()}, tmp)
                tmp.replace(path)
            schedule.step()
            elapsed = time.monotonic() - t0
            writer.writerow([epoch, lr, train_loss, train_acc, val_loss, val_acc, elapsed])
            stream.flush()
            print(f"{epoch:02d}/{cfg['epochs']} train={train_acc:.2%} val={val_acc:.2%} "
                  f"val_loss={val_loss:.4f} ({elapsed:.1f}s)", flush=True)
    metadata["training_seconds"] = time.monotonic() - start
    if augmentation is not None:
        metadata["augmented_examples"] = int(augmentation.applied.item())
        metadata["training_examples_seen"] = training_examples_seen
    write_json(out / "run.json", metadata)


def evaluate(device, audit, manifest, *, out=OUT):
    path = out / "resnet20.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["split_sha256"] != audit["split_sha256"]:
        raise ValueError("Checkpoint split does not match prepared evaluation data")
    model = ResNet20(ckpt["config"].get("width_multiplier", 1)).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state"])
    norm = normalizer(ckpt["mean"], ckpt["std"], device)
    batch_size = ckpt["config"]["batch_size"]
    images, y, heldout = arrays("local_test", manifest, device)
    logits = predict(model, images, norm, batch_size)
    predictions = logits.argmax(1)
    counts = torch.bincount(y * 10 + predictions, minlength=100).reshape(10, 10).cpu().numpy()
    tp = counts.diagonal()
    precision = tp / np.maximum(counts.sum(0), 1)
    recall = tp / np.maximum(counts.sum(1), 1)
    f1 = 2 * tp / np.maximum(counts.sum(0) + counts.sum(1), 1)
    confidence = logits.softmax(1).amax(1).tolist()
    write_rows(out / "local_test_predictions.csv", ["id", "filename", "true_label", "prediction", "confidence"],
               [(r["id"], r["filename"], r["label"], p, c) for r, p, c in zip(heldout, predictions.tolist(), confidence)])
    official = torch.from_numpy(np.load(CACHE / "test.npy")).to(device)
    official_preds = predict(model, official, norm, batch_size).argmax(1).tolist()
    test_ids = [r["id"] for r in rows(DATA / "test.csv")]
    if test_ids != [r["id"] for r in rows(DATA / "sample_submission.csv")] or len(test_ids) != len(official_preds):
        raise ValueError("Competition prediction rows are misaligned")
    write_rows(out / "submission.csv", ["id", "label"], zip(test_ids, official_preds))
    n, correct = int(counts.sum()), int(tp.sum())
    p, z = correct / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / (1 + z*z/n)
    metrics = {
        "checkpoint_sha256": sha256(path), "split_sha256": ckpt["split_sha256"],
        "best_epoch": ckpt["epoch"], "best_validation_accuracy": ckpt["validation_accuracy"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "local_test_examples": n, "local_test_correct": correct, "local_test_accuracy": p,
        "local_test_loss": F.cross_entropy(logits, y).item(), "macro_f1": float(f1.mean()),
        "accuracy_wilson_95": [center-half, center+half], "confusion_matrix_true_by_pred": counts.tolist(),
        "class_precision": precision.tolist(), "class_recall": recall.tolist(), "class_f1": f1.tolist(),
        "competition_predictions": len(official_preds), "competition_accuracy": None,
    }
    write_json(out / "metrics.json", metrics)
    print(f"Local test accuracy: {correct}/{n} = {p:.2%}. Trainable parameters: {metrics['trainable_parameters']:,}.")
    print(f"Competition prediction rows: {len(official_preds)}. Hidden-label accuracy unavailable locally.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    audit = json.loads((OUT / "data_audit.json").read_text())
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Split manifest was changed after auditing")
    manifest = rows(OUT / "splits.csv")
    if args.action == "train":
        train(json.loads((ROOT / "config.json").read_text()), device, audit, manifest)
    else:
        evaluate(device, audit, manifest)


if __name__ == "__main__":
    main()
