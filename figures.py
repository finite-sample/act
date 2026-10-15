"""Frontier plot and paired table from results/runs.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
METRICS = ["acc", "agree", "excess_agree", "soft_agree", "confidence"]
COLORS = {
    "no-aug": "#52514e",
    "aug": "#2a78d6",
    "aug + wd=0.5": "#2a78d6",
    "aug + wd=2": "#2a78d6",
    "ACT lam=0.5": "#eb6834",
    "ACT lam=2": "#eb6834",
    "ACT lam=8": "#eb6834",
    "aug + label smooth 0.2": "#1baf7a",
    "aug + min-entropy 0.5": "#8e5ad0",
}
MARKERS = {
    "no-aug": "s",
    "aug": "o",
    "aug + wd=0.5": "^",
    "aug + wd=2": "v",
    "ACT lam=0.5": "o",
    "ACT lam=2": "^",
    "ACT lam=8": "v",
    "aug + label smooth 0.2": "D",
    "aug + min-entropy 0.5": "D",
}


def paired(df, base="aug"):
    """Mean ± SE over trials of (arm − base), paired on (condition, trial)."""
    wide = df.set_index(["condition", "trial", "arm"])[METRICS].unstack("arm")
    rows = []
    for cond in df.condition.unique():
        w = wide.loc[cond]
        for arm in df.arm.unique():
            if arm == base:
                continue
            row = dict(condition=cond, arm=arm)
            for m in METRICS:
                d = (w[(m, arm)] - w[(m, base)]).dropna()
                row[m] = f"{d.mean():+.3f} ± {d.sem():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def frontier(df, key="condition", base="aug"):
    """Paired change vs `base` in accuracy (x) and hard agreement (y).

    Pairing on trial removes split-to-split variation, which otherwise swamps
    differences between arms. An arm beats the regularisation frontier if it
    sits above the blue points at the same x.
    """
    cells = list(df[key].unique())
    fig, axes = plt.subplots(
        1, len(cells), figsize=(4.6 * len(cells), 4.4), squeeze=False
    )
    for ax, cell in zip(axes[0], cells):
        wide = df[df[key] == cell].set_index(["trial", "arm"])[["acc", "agree"]]
        wide = wide.unstack("arm")
        for arm in COLORS:
            if arm == base or ("acc", arm) not in wide:
                continue
            dx = wide[("acc", arm)] - wide[("acc", base)]
            dy = wide[("agree", arm)] - wide[("agree", base)]
            ax.errorbar(
                dx.mean(), dy.mean(), xerr=2 * dx.sem(), yerr=2 * dy.sem(),
                fmt=MARKERS[arm], color=COLORS[arm], ms=7, lw=1.5, label=arm,
            )  # fmt: skip
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_title(cell)
        ax.set_xlabel(f"Δ test accuracy vs {base}")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel(f"Δ agreement vs {base} (±2 paired SE)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(RESULTS / "frontier.png", dpi=130)
    plt.close(fig)


def compute(df, base="aug"):
    """Compute comparison: paired change vs `base` in accuracy and agreement,
    next to training time, training memory and inference passes."""
    wide = df.pivot_table(
        index=["condition", "trial"], columns="arm", values=["acc", "agree"]
    )
    cost = df.groupby("arm")[["train_s", "train_mem_mb", "infer_passes"]].mean()
    rows = []
    for cond in df.condition.unique():
        w = wide.loc[cond]
        for arm in df.arm.unique():
            if arm == base:
                continue
            row = dict(condition=cond, arm=arm)
            for m in ("acc", "agree"):
                d = w[(m, arm)] - w[(m, base)]
                row[f"d_{m}"], row[f"d_{m}_se"] = d.mean(), d.sem()
            rows.append(row | cost.loc[arm].to_dict())
    t = pd.DataFrame(rows)
    t.round(4).to_csv(RESULTS / "compute_paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    colors = {"ACT lam=0.5": "#eb6834", "ACT k=2 lam=0.5": "#eda100",
              "ensemble 4x1": "#2a78d6", "codistill 2x2": "#1baf7a"}  # fmt: skip
    for ax, cond in zip(axes, df.condition.unique()):
        for _, r in t[t.condition == cond].iterrows():
            ax.errorbar(
                r.train_s, r.d_agree, yerr=2 * r.d_agree_se, fmt="o", ms=7,
                color=colors[r.arm], label=f"{r.arm} ({r.infer_passes:.0f}x inference)",
            )  # fmt: skip
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.axvline(cost.loc[base, "train_s"], color="#c3c2b7", lw=1, ls=":")
        ax.set_title(cond)
        ax.set_xlabel("training seconds per model (M4, MPS)")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"Δ agreement vs {base} (±2 paired SE)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(RESULTS / "compute.png", dpi=130)
    plt.close(fig)
    return t


def main():
    df = pd.read_csv(RESULTS / "runs.csv")
    pd.set_option("display.width", 200)
    print(df.groupby(["condition", "arm"])[METRICS].mean().round(3).to_string())
    print("\nPaired differences vs 'aug' (same pools, same compute):")
    t = paired(df)
    print(t.to_string(index=False))
    t.to_csv(RESULTS / "paired_vs_aug.csv", index=False)
    frontier(df)
    if (RESULTS / "compute.csv").exists():
        t = compute(pd.read_csv(RESULTS / "compute.csv"))
        print("\nCompute comparison (paired vs 'aug'):")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
