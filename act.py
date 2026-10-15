"""Augmentation-consistency training (ACT) and the measurements used to judge it.

Loss, for k augmented views of a batch with logits z_1..z_k:
    mean_i CE(z_i, y)  +  lam * mean_{i<j} 0.5 [KL(p_i || p_j) + KL(p_j || p_i)]
(the README's symmetric form; the original notebook used one direction only).
lam = 0 with k views is plain augmentation at the same compute, which is the
baseline that isolates the consistency term.

Stability follows Turney (1995) / the "prediction churn" literature: train two
models on disjoint samples and measure how often they agree on held-out data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """The original notebook's CIFAR network."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x, gen, pad=4):
    """Standard CIFAR augmentation, independently per image: reflect-pad by
    `pad`, random crop back to size, random horizontal flip."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device="cpu")
    dx = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device="cpu")
    rows = (dy[:, None] + torch.arange(h)).to(x.device)
    cols = (dx[:, None] + torch.arange(w)).to(x.device)
    out = xp[torch.arange(b, device=x.device)[:, None, None, None],
             torch.arange(x.shape[1], device=x.device)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]  # fmt: skip
    flip = (torch.rand(b, generator=gen, device="cpu") < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def symmetric_kl(logits):
    """Mean over view pairs of 0.5 [KL(p_i||p_j) + KL(p_j||p_i)], batch-averaged."""
    logp = [F.log_softmax(z, dim=1) for z in logits]
    total, pairs = 0.0, 0
    for i in range(len(logp)):
        for j in range(i + 1, len(logp)):
            pi, pj = logp[i].exp(), logp[j].exp()
            kl_ij = (pi * (logp[i] - logp[j])).sum(1)
            kl_ji = (pj * (logp[j] - logp[i])).sum(1)
            total = total + 0.5 * (kl_ij + kl_ji).mean()
            pairs += 1
    return total / pairs


def mean_entropy(logits):
    logp = F.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(1).mean()


def act_loss(model, x, y, k, lam, gen, smooth=0.0, ent=0.0, aug=None):
    """CE (optionally label-smoothed) averaged over k views, + lam * view
    disagreement, + ent * prediction entropy. ent > 0 is the minimum-entropy
    churn regulariser of Bhojanapalli et al. (2021); smooth is label smoothing,
    the basis of Bahri & Jiang's (2021) churn method. Views are augmented
    unless aug is False; by default a single view is the raw image."""
    aug = k > 1 if aug is None else aug
    views = [augment(x, gen) for _ in range(k)] if aug else [x] * k
    logits = [model(v) for v in views]
    loss = torch.stack(
        [F.cross_entropy(z, y, label_smoothing=smooth) for z in logits]
    ).mean()
    if ent > 0:
        loss = loss + ent * torch.stack([mean_entropy(z) for z in logits]).mean()
    if lam > 0:
        loss = loss + lam * symmetric_kl(logits)
    return loss


def train(
    X, Y, *, k, lam, weight_decay, epochs, seed, device, smooth=0.0, ent=0.0,
    aug=None, batch=64, lr=1e-3,
):  # fmt: skip
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(n, generator=gen).split(batch):
            opt.zero_grad()
            act_loss(model, X[idx], Y[idx], k, lam, gen, smooth, ent, aug).backward()
            opt.step()
    return model


def codistill_loss(model_a, model_b, x, y, k, alpha, gen):
    """Online co-distillation (Anil et al. 2018), the churn method of
    Bhojanapalli et al. (2021): two models see the same k augmented views; each
    pays CE plus alpha * KL(teacher || itself), the teacher being the other
    model's prediction with gradients stopped."""
    loss = 0.0
    for v in [augment(x, gen) for _ in range(k)]:
        za, zb = model_a(v), model_b(v)
        loss = loss + F.cross_entropy(za, y) + F.cross_entropy(zb, y)
        if alpha > 0:
            lpa, lpb = F.log_softmax(za, 1), F.log_softmax(zb, 1)
            kl_a = F.kl_div(lpa, lpb.detach(), log_target=True, reduction="batchmean")
            kl_b = F.kl_div(lpb, lpa.detach(), log_target=True, reduction="batchmean")
            loss = loss + alpha * (kl_a + kl_b)
    return loss / k


def train_codistill(
    X, Y, *, k, alpha, burn_in, epochs, seed, device, batch=64, lr=1e-3
):  # fmt: skip
    """Train two co-distilled models; alpha switches on after `burn_in` epochs,
    as in Anil et al. Only the first model is returned: inference is 1x."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model_a, model_b = SmallCNN().to(device), SmallCNN().to(device)
    params = list(model_a.parameters()) + list(model_b.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        model_a.train(), model_b.train()
        a = alpha if epoch >= burn_in else 0.0
        for idx in torch.randperm(len(X), generator=gen).split(batch):
            opt.zero_grad()
            codistill_loss(model_a, model_b, X[idx], Y[idx], k, a, gen).backward()
            opt.step()
    return model_a


@torch.no_grad()
def predict_proba(model, X, batch=1000):
    model.eval()
    return torch.cat([F.softmax(model(xb), 1) for xb in X.split(batch)]).cpu().numpy()


def predict_proba_ensemble(models, X):
    """Average of the members' predicted probabilities."""
    return np.mean([predict_proba(m, X) for m in models], axis=0)


def pair_metrics(p1, p2, y, num_classes=10):
    """Accuracy and agreement for two models' test probabilities.

    Hard agreement is mechanically tied to accuracy: it is at least
    a1 + a2 - 1, and if the two models' errors were independent and uniform
    over wrong classes it would be a1 a2 + (1-a1)(1-a2)/(K-1). `excess_agree`
    is agreement above that independence benchmark, i.e. how correlated the
    models are beyond what their accuracies force. `soft_agree` is the
    original exp(-symmetric KL), which also rises when both models are merely
    less confident.
    """
    h1, h2 = p1.argmax(1), p2.argmax(1)
    a1, a2 = (h1 == y).mean(), (h2 == y).mean()
    agree = (h1 == h2).mean()
    indep = a1 * a2 + (1 - a1) * (1 - a2) / (num_classes - 1)
    eps = 1e-12
    lp1, lp2 = np.log(p1 + eps), np.log(p2 + eps)
    skl = 0.5 * ((p1 * (lp1 - lp2)).sum(1) + (p2 * (lp2 - lp1)).sum(1)).mean()
    return dict(
        acc=(a1 + a2) / 2,
        agree=agree,
        agree_floor=max(a1 + a2 - 1, 0.0),
        excess_agree=agree - indep,
        soft_agree=float(np.exp(-skl)),
        confidence=(p1.max(1).mean() + p2.max(1).mean()) / 2,
    )
