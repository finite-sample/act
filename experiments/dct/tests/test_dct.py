import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dct import MLP, dct_loss, pair_metrics, pairwise_prob_mse


def test_pairwise_mse_is_twice_variance_across_masks():
    # Fraternal-dropout identity: mean pairwise squared difference over i.i.d.
    # draws = 2 * variance. Checked on probabilities from real dropout passes.
    torch.manual_seed(0)
    model = MLP(5, dropout=0.3).train()
    x = torch.randn(64, 5)
    logits = [model(x) for _ in range(400)]
    probs = torch.stack([F.softmax(z, 1) for z in logits])
    var = probs.var(0, unbiased=True).mean()
    assert pairwise_prob_mse(logits).item() == pytest.approx(2 * var.item(), rel=0.02)


def test_linear_dropout_variance_is_weighted_ridge():
    # For logit w'(m * x)/(1-q) with m_j ~ Bernoulli(1-q), the variance over
    # masks is q/(1-q) * sum_j w_j^2 x_j^2: a data-weighted L2 penalty.
    rng = np.random.default_rng(0)
    q, w, x = 0.3, rng.normal(size=6), rng.normal(size=6)
    masks = rng.random((200_000, 6)) > q
    logits = (masks * x) @ w / (1 - q)
    assert logits.var() == pytest.approx(q / (1 - q) * np.sum(w**2 * x**2), rel=0.02)


def test_lam_zero_is_mean_ce_over_passes():
    torch.manual_seed(0)
    model = MLP(4, dropout=0.0)
    x, y = torch.randn(10, 4), torch.randint(0, 2, (10,))
    assert dct_loss(model, x, y, 3, 0.0).item() == pytest.approx(
        F.cross_entropy(model(x), y).item(), rel=1e-6
    )


def test_pair_metrics_hand_example():
    y = np.array([0, 1, 1, 0])
    p1 = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    p2 = np.array([[0.8, 0.2], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6]])
    m = pair_metrics(p1, p2, y)
    assert m["acc"] == pytest.approx((0.75 + 0.75) / 2)
    assert m["agree"] == pytest.approx(0.5)
    assert m["excess_agree"] == pytest.approx(0.5 - (0.75**2 + 0.25**2))
