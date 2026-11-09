"""Train one or more prespecified augmentation arms on the frozen baseline split."""
import argparse
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from prepare_data import OUT, ROOT, sha256
from model import ResNet20
from run_baseline import arrays, evaluate, normalize, normalizer, predict, rows, train, write_rows


RESULTS = ROOT / "output" / "augmentations"
SWAPS = ROOT / ".cache" / "augmentations"
ARMS = {
    "A0_none": {},
    "A1_flip": {"direction": "horizontal"},
    "A2_rotation": {"max_degrees": 10},
    "A3_color": {"max_factor_offset": 0.1, "order": ["brightness", "contrast", "saturation"]},
    "A4_resolution": {"downsample_size": 28},
    "A5_noise": {"noise_std": 0.02},
    "A6_background": {"minimum_mask_iou": 0.75},
}


class BatchAugmentation:
    def __init__(self, name, device, seed, split_hash):
        self.name = name
        self.params = ARMS[name]
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.applied = torch.zeros((), dtype=torch.int64, device=device)
        if name == "A6_background":
            report = json.loads((RESULTS / "background_qa" / "pool.json").read_text())
            if report["split_sha256"] != split_hash or report["minimum_mask_iou"] != self.params["minimum_mask_iou"]:
                raise ValueError("Background pool and training protocol do not match")
            for filename, digest in report["cache_sha256"].items():
                if sha256(SWAPS / filename) != digest:
                    raise ValueError(f"Background pool mismatch: {filename}")
            self.swaps = torch.from_numpy(np.load(SWAPS / "swaps.npy")).to(device)
            self.valid = torch.from_numpy(np.load(SWAPS / "valid.npy")).to(device)
            self.pool_sha256 = report["cache_sha256"]

    def __call__(self, images, idx, norm):
        raw = images[idx]
        if self.name == "A0_none":
            return normalize(raw, *norm)
        mask = torch.rand(len(idx), device=idx.device, generator=self.generator) < 0.5
        if self.name == "A6_background":
            mask = mask & self.valid[idx]
            self.applied += mask.sum()
            return normalize(torch.where(mask[:, None, None, None], self.swaps[idx], raw), *norm)
        self.applied += mask.sum()
        if self.name == "A1_flip":
            return normalize(torch.where(mask[:, None, None, None], raw.flip((2,)), raw), *norm)

        x = raw.permute(0, 3, 1, 2).float().div_(255)
        if self.name == "A2_rotation":
            limit = math.radians(self.params["max_degrees"])
            angle = torch.empty(len(idx), device=x.device).uniform_(-limit, limit, generator=self.generator)
            c, s, zero = angle.cos(), angle.sin(), torch.zeros_like(angle)
            matrix = torch.stack((c, -s, zero, s, c, zero), dim=1).view(-1, 2, 3)
            grid = F.affine_grid(matrix, x.shape, align_corners=False)
            changed = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        elif self.name == "A3_color":
            offset = self.params["max_factor_offset"]
            b, c, s = torch.empty(3, len(idx), 1, 1, 1, device=x.device).uniform_(
                1 - offset, 1 + offset, generator=self.generator)
            weights = x.new_tensor([0.2989, 0.587, 0.114])[None, :, None, None]
            changed = (x * b).clamp_(0, 1)
            avg = (changed * weights).sum(1, keepdim=True).mean((2, 3), keepdim=True)
            changed = (avg + c * (changed - avg)).clamp_(0, 1)
            gray = (changed * weights).sum(1, keepdim=True)
            changed = (gray + s * (changed - gray)).clamp_(0, 1)
        elif self.name == "A4_resolution":
            small = F.interpolate(x, size=(self.params["downsample_size"],) * 2, mode="bilinear",
                                  align_corners=False, antialias=True)
            changed = F.interpolate(small, size=(32, 32), mode="bilinear", align_corners=False)
        elif self.name == "A5_noise":
            noise = torch.randn(x.shape, device=x.device, generator=self.generator)
            changed = (x + self.params["noise_std"] * noise).clamp_(0, 1)
        else:
            raise ValueError(f"Unknown augmentation: {self.name}")
        x = torch.where(mask[:, None, None, None], changed, x)
        return ((x - norm[0]) / norm[1]).contiguous(memory_format=torch.channels_last)


def compare(device, audit, manifest):
    """Independently verify the best checkpoints on validation and assemble the results."""
    val, labels, _ = arrays("validation", manifest, device)
    model = ResNet20().to(device=device, memory_format=torch.channels_last)
    table = []
    for name, folder in [("B_baseline", OUT), *((arm, RESULTS / arm) for arm in ARMS)]:
        run = json.loads((folder / "run.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        ckpt_path = folder / "resnet20.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        history = rows(folder / "history.csv")
        best = max(history, key=lambda row: float(row["validation_accuracy"]))
        if (len(history) != run["config"]["epochs"] or ckpt["epoch"] != int(best["epoch"])
                or ckpt["split_sha256"] != audit["split_sha256"]
                or sha256(ckpt_path) != metrics["checkpoint_sha256"]):
            raise ValueError(f"Run integrity check failed: {name}")
        model.load_state_dict(ckpt["model_state"])
        norm = normalizer(ckpt["mean"], ckpt["std"], device)
        logits = predict(model, val, norm, run["config"]["batch_size"])
        score = int((logits.argmax(1) == labels).sum()) / len(labels)
        if not (score == float(best["validation_accuracy"]) == ckpt["validation_accuracy"]
                == metrics["best_validation_accuracy"]):
            raise ValueError(f"Validation score mismatch: {name}")
        table.append((name, ckpt["epoch"], score, metrics["local_test_accuracy"],
                      metrics["macro_f1"], run["training_seconds"]))
    base, control = table[0][2], table[1][2]
    write_rows(RESULTS / "comparison.csv",
               ["arm", "best_epoch", "validation_accuracy_pct", "delta_vs_A0_pp", "delta_vs_B_pp",
                "local_test_accuracy_pct", "local_test_macro_f1", "training_seconds"],
               [(name, epoch, round(100 * val_acc, 2), round(100 * (val_acc - control), 2),
                 round(100 * (val_acc - base), 2), round(100 * test, 2), round(f1, 6), round(sec, 2))
                for name, epoch, val_acc, test, f1, sec in table])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "compare"))
    parser.add_argument("arms", nargs="*", metavar="ARM")
    args = parser.parse_args()
    if (args.action == "compare") == bool(args.arms) or any(arm not in ARMS for arm in args.arms):
        parser.error(f"train/evaluate require arms from {', '.join(ARMS)}; compare takes no arms")
    audit = json.loads((OUT / "data_audit.json").read_text())
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Changed split manifest")
    manifest = rows(OUT / "splits.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action == "compare":
        compare(device, audit, manifest)
        return
    for name in args.arms:
        out = RESULTS / name
        if args.action == "evaluate":
            evaluate(device, audit, manifest, out=out)
            continue
        cfg = json.loads((ROOT / "config.json").read_text())
        augmentation = BatchAugmentation(name, device, cfg["seed"], audit["split_sha256"])
        cfg.update(crop_padding=0, horizontal_flip_probability=0.5 if name == "A1_flip" else 0.0,
                   augmentation={"name": name, "probability": 0.0 if name == "A0_none" else 0.5,
                                 **ARMS[name]})
        if name == "A6_background":
            cfg["augmentation"]["cache_sha256"] = augmentation.pool_sha256
        if (out / "run.json").exists():
            raise FileExistsError(f"Existing training run in {out}")
        out.mkdir(parents=True, exist_ok=True)
        print(f"Training {name}", flush=True)
        train(cfg, device, audit, manifest, out=out, augmentation=augmentation)


if __name__ == "__main__":
    main()
