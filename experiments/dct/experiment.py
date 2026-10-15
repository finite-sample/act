"""Stability-accuracy frontier for DCT vs matched-compute and regularised baselines.

Per trial (as in the notebook): hold out 30% as a test set, split the rest
into two disjoint halves, train one model per half with the same method, and
score the pair on the test set. All arms see the same splits for a trial.

Arms:
  standard          1 pass, dropout 0.3 (the notebook's baseline)
  5 passes          5 passes, CE averaged, no penalty (DCT's compute)
  DCT lam=l         5 passes + l * pairwise MSE between pass probabilities
  ... CE on pass 1  the notebook's variant (README averages CE over passes)
  wd=w              standard + AdamW weight decay w   (frontier)
  dropout=0.5       standard with a higher dropout rate (frontier)

Each dataset's results go to results/<dataset>.csv; finished datasets are
skipped on rerun.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dct import pair_metrics, predict_proba, train

EPOCHS = 30
N_JOBS = -1
ARMS = {
    "standard": dict(n_passes=1, lam=0.0, weight_decay=0.0, dropout=0.3),
    "5 passes": dict(n_passes=5, lam=0.0, weight_decay=0.0, dropout=0.3),
    "DCT lam=0.5": dict(n_passes=5, lam=0.5, weight_decay=0.0, dropout=0.3),
    "DCT lam=1.5": dict(n_passes=5, lam=1.5, weight_decay=0.0, dropout=0.3),
    "DCT lam=5": dict(n_passes=5, lam=5.0, weight_decay=0.0, dropout=0.3),
    "DCT lam=1.5, CE on pass 1 (notebook)": dict(
        n_passes=5, lam=1.5, weight_decay=0.0, dropout=0.3, ce_first_only=True
    ),
    "wd=0.1": dict(n_passes=1, lam=0.0, weight_decay=0.1, dropout=0.3),
    "wd=1": dict(n_passes=1, lam=0.0, weight_decay=1.0, dropout=0.3),
    "dropout=0.5": dict(n_passes=1, lam=0.0, weight_decay=0.0, dropout=0.5),
}


def datasets():
    """The notebook's four datasets plus UCI Adult (claimed in the README)."""
    out = {
        "High Noise": make_classification(
            n_samples=700, n_features=15, n_informative=8, n_redundant=3,
            flip_y=0.15, class_sep=0.65, random_state=42,
        ),
        "Small Sample": make_classification(
            n_samples=400, n_features=12, n_informative=8, n_redundant=2,
            flip_y=0.05, class_sep=0.8, random_state=43,
        ),
        "High Dim": make_classification(
            n_samples=600, n_features=40, n_informative=8, n_redundant=5,
            flip_y=0.08, class_sep=0.7, random_state=44,
        ),
        "Breast Cancer": load_breast_cancer(return_X_y=True),
    }  # fmt: skip
    adult = fetch_openml("adult", version=2, as_frame=True)
    X, y = adult.data, (adult.target == ">50K").astype(int).to_numpy()
    cats = X.select_dtypes("category").columns
    enc = ColumnTransformer(
        [("cat", OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist",
                               sparse_output=False), cats)],
        remainder="passthrough",
    )  # fmt: skip
    # A 3000-row subsample keeps Adult at the scale of the other datasets.
    idx = np.random.default_rng(0).choice(len(X), 3000, replace=False)
    out["Adult (3k)"] = (enc.fit_transform(X.iloc[idx]).astype(float), y[idx])
    return out


def one_trial(X, y, trial):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=trial)
    X1, X2, y1, y2 = train_test_split(Xtr, ytr, test_size=0.5, random_state=trial * 2)
    # Scale on each half's own data: the two models must not share information.
    halves = []
    for Xh, yh in ((X1, y1), (X2, y2)):
        sc = StandardScaler().fit(Xh)
        halves.append((sc, torch.tensor(sc.transform(Xh)).float(), torch.tensor(yh)))
    torch.set_num_threads(1)
    rows = []
    for arm, cfg in ARMS.items():
        probs = []
        for h, (sc, Xh, yh) in enumerate(halves):
            model = train(Xh, yh, **cfg, epochs=EPOCHS, seed=1000 * trial + h)
            probs.append(predict_proba(model, torch.tensor(sc.transform(Xte)).float()))
        rows.append(dict(arm=arm, trial=trial) | pair_metrics(*probs, yte))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    for name, (X, y) in datasets().items():
        path = args.out / f"{name}.csv"
        if path.exists():
            continue
        res = Parallel(n_jobs=N_JOBS)(
            delayed(one_trial)(X, y, t) for t in range(args.trials)
        )
        df = pd.DataFrame([r for rows in res for r in rows])
        df.insert(0, "dataset", name)
        df.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    # Importable-module entry so joblib pickles workers by reference.
    import experiment

    experiment.main()
