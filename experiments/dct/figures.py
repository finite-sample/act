"""Frontier plots and paired tables from results/<dataset>.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
METRICS = ["acc", "agree", "excess_agree", "soft_agree", "confidence"]
COLORS = {
    "standard": "#52514e",
    "5 passes": "#2a78d6",
    "wd=0.1": "#2a78d6",
    "wd=1": "#2a78d6",
    "dropout=0.5": "#2a78d6",
    "DCT lam=0.5": "#eb6834",
    "DCT lam=1.5": "#eb6834",
    "DCT lam=5": "#eb6834",
    "DCT lam=1.5, CE on pass 1 (notebook)": "#1baf7a",
}
MARKERS = {
    "standard": "s",
    "5 passes": "o",
    "wd=0.1": "^",
    "wd=1": "v",
    "dropout=0.5": "D",
    "DCT lam=0.5": "o",
    "DCT lam=1.5": "^",
    "DCT lam=5": "v",
    "DCT lam=1.5, CE on pass 1 (notebook)": "D",
}


def paired(df, base="standard"):
    """Mean ± SE over trials of (arm − base), paired on (dataset, trial)."""
    wide = df.set_index(["dataset", "trial", "arm"])[METRICS].unstack("arm")
    rows = []
    for cond in df.dataset.unique():
        w = wide.loc[cond]
        for arm in df.arm.unique():
            if arm == base:
                continue
            row = dict(dataset=cond, arm=arm)
            for m in METRICS:
                d = (w[(m, arm)] - w[(m, base)]).dropna()
                row[m] = f"{d.mean():+.3f} ± {d.sem():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def frontier(df, key="dataset", base="standard"):
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


def main():
    files = sorted(f for f in RESULTS.glob("*.csv") if f.stem != "paired_vs_standard")
    df = pd.concat(pd.read_csv(f) for f in files)
    pd.set_option("display.width", 200)
    print(df.groupby(["dataset", "arm"])[METRICS].mean().round(3).to_string())
    print("\nPaired differences vs 'standard' (same splits):")
    t = paired(df)
    print(t.to_string(index=False))
    t.to_csv(RESULTS / "paired_vs_standard.csv", index=False)
    frontier(df)


if __name__ == "__main__":
    main()
