"""Stability-accuracy frontier for ACT vs plain augmentation and weight decay.

Per trial: draw a pool of 2 * N_HALF CIFAR-10 training images (optionally
with symmetric label noise), split it into two disjoint halves, train one
model per half, and score the pair on the full 10k CIFAR-10 test set (clean
labels). Every arm sees the same pools for a given trial, so arms are paired.

Arms:
  no-aug            1 view, the original notebook's baseline
  aug               k = 4 augmented views, CE only (same compute as ACT)
  aug + wd=w        same, AdamW weight decay w: the generic-regularisation frontier
  ACT lam=l         k = 4 views, CE + l * symmetric KL between views
  aug + label smooth / min-entropy
                    the churn literature's own regularisers (Bahri & Jiang
                    2021; Bhojanapalli et al. 2021), same views and compute

Each (condition, arm, trial) result is cached on disk by joblib.Memory, so an
interrupted run resumes where it stopped; results/runs.csv is written from
the cache at the end.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from joblib import Memory

from act import (
    pair_metrics,
    predict_proba,
    predict_proba_ensemble,
    train,
    train_codistill,
)

N_HALF = 2000
EPOCHS = 30
K = 4
ARMS = {
    "no-aug": dict(k=1, lam=0.0, weight_decay=0.0),
    "aug": dict(k=K, lam=0.0, weight_decay=0.0),
    "aug + wd=0.5": dict(k=K, lam=0.0, weight_decay=0.5),
    "aug + wd=2": dict(k=K, lam=0.0, weight_decay=2.0),
    "ACT lam=0.5": dict(k=K, lam=0.5, weight_decay=0.0),
    "ACT lam=2": dict(k=K, lam=2.0, weight_decay=0.0),
    "ACT lam=8": dict(k=K, lam=8.0, weight_decay=0.0),
    "aug + label smooth 0.2": dict(k=K, lam=0.0, weight_decay=0.0, smooth=0.2),
    "aug + min-entropy 0.5": dict(k=K, lam=0.0, weight_decay=0.0, ent=0.5),
}
CONDITIONS = {"clean": 0.0, "20% label noise": 0.2}
# Compute comparison (--extra). All but "ACT k=2" make 4 forward passes per
# training step; they differ in how many models predict at test time.
EXTRA_ARMS = {
    "aug": dict(kind="single", infer=1, **ARMS["aug"]),
    "ACT lam=0.5": dict(kind="single", infer=1, **ARMS["ACT lam=0.5"]),
    "ACT k=2 lam=0.5": dict(kind="single", infer=1, k=2, lam=0.5, weight_decay=0.0),
    "ensemble 4x1": dict(kind="ensemble", infer=4, m=4),
    "codistill 2x2": dict(kind="codistill", infer=1, k=2, alpha=1.0, burn_in=5),
}
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
memory = Memory("cache", verbose=0)


def load_cifar():
    """CIFAR-10 as normalised float tensors ((X_train, y_train), (X_test, y_test)).

    From the Hugging Face mirror (uoft-cs/cifar10): the Toronto host that
    torchvision downloads from serves ~50 KB/s.
    """
    ds = load_dataset("uoft-cs/cifar10").with_format("numpy")

    def split(name):
        b = ds[name][:]
        x = torch.tensor(b["img"]).permute(0, 3, 1, 2).float() / 255
        x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        return x, torch.tensor(b["label"])

    return split("train"), split("test")


def make_pool(y_train, trial, noise):
    """Indices of two disjoint halves, and (possibly noisy) labels for them."""
    rng = np.random.default_rng(trial)
    idx = rng.choice(len(y_train), 2 * N_HALF, replace=False)
    labels = y_train[idx].clone()
    flip = rng.random(len(idx)) < noise
    # Symmetric noise: a flipped label goes to a uniformly chosen *other* class.
    labels[flip] = torch.tensor(
        (labels[flip].numpy() + rng.integers(1, 10, flip.sum())) % 10
    )
    return idx[:N_HALF], idx[N_HALF:], labels[:N_HALF], labels[N_HALF:]


@memory.cache
def run_pair(cond, arm, trial):
    """Train one model per half of the trial's pool; metrics on the test set."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (Xtr, ytr), (Xte, yte) = load_cifar()
    halves = make_pool(ytr, trial, CONDITIONS[cond])
    probs = []
    for half, (idx, lab) in enumerate([halves[0::2], halves[1::2]]):
        model = train(
            Xtr[idx].to(device), lab.to(device), **ARMS[arm], epochs=EPOCHS,
            seed=1000 * trial + half, device=device,
        )  # fmt: skip
        probs.append(predict_proba(model, Xte.to(device)))
    return {k: float(v) for k, v in pair_metrics(*probs, yte.numpy()).items()}


def fit_predictor(spec, X, Y, seed, device):
    """Train the arm's model(s) on one half; return a predict function."""
    if spec["kind"] == "single":
        cfg = {k: v for k, v in spec.items() if k not in ("kind", "infer")}
        model = train(X, Y, **cfg, epochs=EPOCHS, seed=seed, device=device)
        return lambda Xte: predict_proba(model, Xte)
    if spec["kind"] == "ensemble":
        models = [
            train(
                X, Y, k=1, lam=0.0, weight_decay=0.0, aug=True, epochs=EPOCHS,
                seed=seed + 100 * j, device=device,
            )  # fmt: skip
            for j in range(spec["m"])
        ]
        return lambda Xte: predict_proba_ensemble(models, Xte)
    model = train_codistill(
        X, Y, k=spec["k"], alpha=spec["alpha"], burn_in=spec["burn_in"],
        epochs=EPOCHS, seed=seed, device=device,
    )  # fmt: skip
    return lambda Xte: predict_proba(model, Xte)


@memory.cache
def run_extra(cond, arm, trial):
    """As run_pair, plus training wall-clock and MPS memory per half.

    Memory is the driver allocation after training minus before. MPS keeps
    freed blocks cached, so after empty_cache() this approximates the peak."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (Xtr, ytr), (Xte, yte) = load_cifar()
    halves = make_pool(ytr, trial, CONDITIONS[cond])
    Xte = Xte.to(device)
    probs, secs, mem = [], [], []
    for half, (idx, lab) in enumerate([halves[0::2], halves[1::2]]):
        X, Y = Xtr[idx].to(device), lab.to(device)
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
            base = torch.mps.driver_allocated_memory()
        t0 = time.perf_counter()
        predict = fit_predictor(EXTRA_ARMS[arm], X, Y, 1000 * trial + half, device)
        if device == "mps":
            torch.mps.synchronize()
            mem.append((torch.mps.driver_allocated_memory() - base) / 2**20)
        secs.append(time.perf_counter() - t0)
        probs.append(predict(Xte))
    m = {k: float(v) for k, v in pair_metrics(*probs, yte.numpy()).items()}
    return m | dict(
        train_s=float(np.mean(secs)),
        train_mem_mb=float(np.max(mem)) if mem else float("nan"),
        infer_passes=EXTRA_ARMS[arm]["infer"],
    )


def main_extra(trials, out):
    rows = []
    for trial in range(trials):
        for cond in CONDITIONS:
            for arm in EXTRA_ARMS:
                m = run_extra(cond, arm, trial)
                rows.append(dict(condition=cond, arm=arm, trial=trial) | m)
                msg = f"acc={m['acc']:.3f} agree={m['agree']:.3f} {m['train_s']:.0f}s"
                print(f"{cond:16s} {arm:16s} trial {trial}: {msg}", flush=True)
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/runs.csv"))
    ap.add_argument(
        "--extra", action="store_true",
        help="compute comparison (ensemble, co-distillation) -> results/compute.csv",
    )  # fmt: skip
    args = ap.parse_args()
    if args.extra:
        return main_extra(args.trials, Path("results/compute.csv"))
    rows = []
    for trial in range(args.trials):
        for cond in CONDITIONS:
            for arm in ARMS:
                m = run_pair(cond, arm, trial)
                rows.append(dict(condition=cond, arm=arm, trial=trial) | m)
                msg = f"acc={m['acc']:.3f} agree={m['agree']:.3f}"
                print(f"{cond:16s} {arm:22s} trial {trial}: {msg}", flush=True)
    args.out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
