"""Dropout-consistency training (DCT) and the measurements used to judge it.

Loss for n stochastic forward passes z_1..z_n of the same batch (dropout on):
    mean_i CE(z_i, y) + lam * mean_{i<j} MSE(softmax z_i, softmax z_j)
as in the README (the notebook applied CE to the first pass only). With
lam = 0 this is n-pass training at the same compute: the baseline that
isolates the penalty.

Fraternal dropout (Zolna et al. 2018) shows the pairwise penalty is twice the
variance of the predictions across dropout masks; for a linear layer that
variance is a data-weighted ridge penalty (Wager, Wang & Liang 2013). So the
natural yardstick is ordinary regularisation, swept to trace a frontier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The original notebook's network."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def pairwise_prob_mse(logits):
    probs = [F.softmax(z, dim=1) for z in logits]
    pairs = [
        F.mse_loss(probs[i], probs[j])
        for i in range(len(probs))
        for j in range(i + 1, len(probs))
    ]
    return torch.stack(pairs).mean()


def dct_loss(model, x, y, n_passes, lam, ce_first_only=False):
    logits = [model(x) for _ in range(n_passes)]
    if ce_first_only:  # the notebook's variant; the README averages CE
        ce = F.cross_entropy(logits[0], y)
    else:
        ce = torch.stack([F.cross_entropy(z, y) for z in logits]).mean()
    if lam > 0 and n_passes > 1:
        return ce + lam * pairwise_prob_mse(logits)
    return ce


def train(
    X, y, *, n_passes, lam, weight_decay, dropout, epochs, seed, ce_first_only=False,
    lr=2e-3,
):  # fmt: skip
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP(X.shape[1], output_dim=int(y.max()) + 1, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X), generator=gen).split(32):
            opt.zero_grad()
            dct_loss(model, X[idx], y[idx], n_passes, lam, ce_first_only).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return F.softmax(model(X), 1).numpy()


def pair_metrics(p1, p2, y):
    """Accuracy and agreement of two models trained on disjoint samples.

    `excess_agree` is hard agreement above what the two accuracies would give
    if the models erred independently; `soft_agree` is the original
    exp(-symmetric KL), which also rises when both models are just less
    confident, so `confidence` is reported alongside it.
    """
    k = p1.shape[1]
    h1, h2 = p1.argmax(1), p2.argmax(1)
    a1, a2 = (h1 == y).mean(), (h2 == y).mean()
    agree = (h1 == h2).mean()
    indep = a1 * a2 + (1 - a1) * (1 - a2) / (k - 1)
    eps = 1e-12
    lp1, lp2 = np.log(p1 + eps), np.log(p2 + eps)
    skl = 0.5 * ((p1 * (lp1 - lp2)).sum(1) + (p2 * (lp2 - lp1)).sum(1)).mean()
    return dict(
        acc=(a1 + a2) / 2,
        agree=agree,
        excess_agree=agree - indep,
        soft_agree=float(np.exp(-skl)),
        confidence=(p1.max(1).mean() + p2.max(1).mean()) / 2,
    )
