"""Fixed 240-epoch training loop with atomic, resumable checkpoints."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import hashlib
import json
import random
import time
import platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import runtime
import training as operators

HERE = Path(".")
DATA = Path(".")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def status(phase, **kw):
    runtime.atomic_json(
        HERE / "status.json",
        dict(phase=phase, pid=os.getpid(), updated_at_utc=runtime.utc_now(), **kw),
    )


def save(p, value):
    temp = p.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    if p.exists():
        os.replace(p, p.with_name("previous.pt"))
    os.replace(temp, p)


def write_csv(p, values):
    temp = p.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)
    os.replace(temp, p)


def run(config, cache, labels, output_dir, data_dir, stop_after_epoch=None):
    global HERE, DATA
    HERE = output_dir
    DATA = data_dir
    runtime.OUTPUT = output_dir
    torch.set_num_threads(4)
    cfg = config["recipe"]
    config_sha = sha(HERE / "config.json")
    status(
        "queued",
        epochs=cfg["epochs"],
        configuration=config["configuration"],
        policy=config["policy"],
    )
    with runtime.gpu_lock("confirmation", "final_full50k_seed42"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        images = torch.from_numpy(cache).cuda()
        y = torch.from_numpy(labels).cuda()
        norm = [
            torch.tensor(config[k], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
            for k in ["mean", "std"]
        ]
        net = operators.build(cfg).cuda().to(memory_format=torch.channels_last)
        assert (
            sum((p.numel() for p in net.parameters()))
            == config["parameters"]
            <= 5000000
        )
        opt = torch.optim.SGD(
            net.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
        sch = operators.make_scheduler(opt, cfg)
        torch.manual_seed(30042)
        torch.cuda.manual_seed_all(30042)
        history = []
        if (HERE / "last.pt").exists():
            cp = torch.load(HERE / "last.pt", map_location="cpu", weights_only=False)
            assert cp["config_sha256"] == config_sha
            net.load_state_dict(cp["model_state"])
            opt.load_state_dict(cp["optimizer_state"])
            sch.load_state_dict(cp["scheduler_state"])
            history = cp["history"]
            random.setstate(cp["python_rng"])
            np.random.set_state(cp["numpy_rng"])
            torch.set_rng_state(cp["torch_rng"])
            torch.cuda.set_rng_state_all(cp["cuda_rng"])
            del cp
        runtime.atomic_json(
            HERE / "environment.json",
            dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                numpy=np.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(),
                bf16_training=True,
                fp32_inference=True,
            ),
        )
        for epoch in range(len(history) + 1, cfg["epochs"] + 1):
            runtime.check_control("confirmation")
            torch.cuda.synchronize()
            start = time.monotonic()
            order_np, counts, swaps = operators.sampler(
                labels, 42, epoch, cfg["batch_size"]
            )
            flip, angle, cut = operators.augmentation_plan(42, epoch, 1, n=50000)
            order = torch.from_numpy(order_np).cuda()
            fp, ang, cu = [
                torch.as_tensor(v, device="cuda") for v in (flip, angle, cut)
            ]
            net.train()
            totals = torch.zeros(2, dtype=torch.float64, device="cuda")
            lr = opt.param_groups[0]["lr"]
            for pos in range(0, 50000, cfg["batch_size"]):
                ids = order[pos : pos + cfg["batch_size"]]
                x = operators.augment(
                    images, ids, norm, fp[0, ids], ang[0, ids], cu[0, ids], cfg
                )
                opt.zero_grad(set_to_none=True)
                loss, logits = operators.objective(net, x, y[ids], cfg)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                opt.step()
                totals[0] += loss.detach() * len(ids)
                totals[1] += (logits.detach().argmax(1) == y[ids]).sum()
            sch.step()
            torch.cuda.synchronize()
            values = (totals / 50000).tolist()
            history.append(
                dict(
                    epoch=epoch,
                    lr=lr,
                    train_loss=values[0],
                    train_accuracy=values[1],
                    examples=50000,
                    updates=len(counts),
                    seconds=time.monotonic() - start,
                    order_sha256=hashlib.sha256(order_np.tobytes()).hexdigest(),
                    sampler_swaps=swaps,
                )
            )
            save(
                HERE / "last.pt",
                dict(
                    config_sha256=config_sha,
                    epoch=epoch,
                    model_state=net.state_dict(),
                    optimizer_state=opt.state_dict(),
                    scheduler_state=sch.state_dict(),
                    history=history,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                ),
            )
            write_csv(HERE / "history.csv", history)
            status("training", epochs=cfg["epochs"], **history[-1])
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(history[-1]), flush=True)
            if (
                stop_after_epoch is not None
                and epoch >= stop_after_epoch
                and epoch < cfg["epochs"]
            ):
                status("paused_for_reproduction_check", epoch=epoch)
                return
        runtime.check_control("confirmation")
        net.eval()
        save(
            HERE / "model.pt",
            dict(
                model_state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                recipe=cfg,
                policy=config["policy"],
                mean=config["mean"],
                std=config["std"],
                seed=42,
                epoch=cfg["epochs"],
                parameters=config["parameters"],
                training_images=50000,
                config_sha256=config_sha,
            ),
        )
        test = rows(DATA / "test.csv")
        sample = rows(DATA / "sample_submission.csv")
        assert [r["id"] for r in test] == [r["id"] for r in sample] and len(
            test
        ) == 2000
        pixels = []
        for r in test:
            with Image.open(DATA / "test_images" / r["filename"]) as im:
                assert im.mode == "RGB" and im.size == (32, 32)
                pixels.append(np.array(im))
        test_images = torch.from_numpy(np.stack(pixels)).cuda()
        raw = operators.predict(net, test_images, norm)
        scores = raw
        if config["policy"] == "tta":
            scores = 0.5 * (
                raw.softmax(1)
                + operators.predict(net, test_images.flip(2), norm).softmax(1)
            )
        pred = scores.argmax(1).cpu().tolist()
        write_csv(
            HERE / "submission.csv",
            [dict(id=r["id"], label=p) for r, p in zip(test, pred)],
        )
        np.save(HERE / "competition_scores.npy", scores.cpu().numpy())
        runtime.atomic_json(
            HERE / "RESULT.json",
            dict(
                completed_at_utc=runtime.utc_now(),
                configuration=config["configuration"],
                policy=config["policy"],
                epoch=cfg["epochs"],
                parameters=config["parameters"],
                training_images=50000,
                optimizer_updates=sum(row["updates"] for row in history),
                total_training_presentations=50000 * cfg["epochs"],
                validation_accuracy=None,
                kaggle_score=None,
                uploaded=False,
                artifact_sha256={
                    n: sha(HERE / n)
                    for n in [
                        "model.pt",
                        "submission.csv",
                        "competition_scores.npy",
                        "history.csv",
                        "config.json",
                    ]
                },
            ),
        )
        status("awaiting_independent_replay", epoch=cfg["epochs"], epochs=cfg["epochs"])
