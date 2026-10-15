import numpy as np
import pytest
import torch
import torch.nn.functional as F

from act import (
    SmallCNN,
    act_loss,
    augment,
    codistill_loss,
    pair_metrics,
    predict_proba,
    predict_proba_ensemble,
    symmetric_kl,
)


def test_symmetric_kl_matches_torch_kl_div():
    torch.manual_seed(0)
    a, b = torch.randn(8, 10), torch.randn(8, 10)
    ref = 0.5 * (
        F.kl_div(F.log_softmax(b, 1), F.softmax(a, 1), reduction="batchmean")
        + F.kl_div(F.log_softmax(a, 1), F.softmax(b, 1), reduction="batchmean")
    )
    assert symmetric_kl([a, b]).item() == pytest.approx(ref.item(), rel=1e-5)
    assert symmetric_kl([a, a]).item() == pytest.approx(0.0, abs=1e-7)


def test_augment_is_crop_and_flip_of_padded_image():
    gen = torch.Generator().manual_seed(0)
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    out = augment(x, gen, pad=2)
    assert out.shape == x.shape
    xp = F.pad(x, (2, 2, 2, 2), mode="reflect")
    for i in range(2):
        hits = [
            (dy, dx, flip)
            for dy in range(5)
            for dx in range(5)
            for flip in (False, True)
            if torch.equal(
                out[i],
                (lambda c: c.flip(-1) if flip else c)(
                    xp[i, :, dy : dy + 8, dx : dx + 8]
                ),
            )
        ]
        assert hits, "each output must be some crop (optionally flipped) of the input"


def test_lam_zero_equals_mean_ce_over_views():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 10))
    x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))
    l0 = act_loss(model, x, y, k=3, lam=0.0, gen=torch.Generator().manual_seed(1))
    gen = torch.Generator().manual_seed(1)
    ref = torch.stack([F.cross_entropy(model(augment(x, gen)), y) for _ in range(3)])
    assert l0.item() == pytest.approx(ref.mean().item(), rel=1e-6)


def test_pair_metrics_hand_example():
    y = np.array([0, 1, 2, 3])
    p1 = np.eye(10)[[0, 1, 2, 4]] * 0.9 + 0.01
    p2 = np.eye(10)[[0, 1, 5, 4]] * 0.9 + 0.01
    m = pair_metrics(p1, p2, y)
    assert m["acc"] == pytest.approx((0.75 + 0.5) / 2)
    assert m["agree"] == pytest.approx(0.75)
    assert m["agree_floor"] == pytest.approx(0.25)
    indep = 0.75 * 0.5 + 0.25 * 0.5 / 9
    assert m["excess_agree"] == pytest.approx(0.75 - indep)


def test_single_augmented_view_differs_from_raw():
    torch.manual_seed(0)
    model, x, y = (
        SmallCNN().eval(),
        torch.randn(16, 3, 32, 32),
        torch.randint(0, 10, (16,)),
    )
    raw = act_loss(model, x, y, 1, 0.0, torch.Generator().manual_seed(0))
    unaug = act_loss(model, x, y, 1, 0.0, torch.Generator().manual_seed(0), aug=False)
    aug = act_loss(model, x, y, 1, 0.0, torch.Generator().manual_seed(0), aug=True)
    gen = torch.Generator().manual_seed(0)
    ref = F.cross_entropy(model(augment(x, gen)), y)
    assert (
        raw.item() == unaug.item() == pytest.approx(F.cross_entropy(model(x), y).item())
    )
    assert aug.item() == pytest.approx(ref.item())
    assert aug.item() != pytest.approx(raw.item())


def test_codistill_alpha0_is_two_independent_ce_losses():
    torch.manual_seed(0)
    a, b = SmallCNN().eval(), SmallCNN().eval()
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    loss = codistill_loss(a, b, x, y, 2, 0.0, torch.Generator().manual_seed(1))
    gen = torch.Generator().manual_seed(1)
    views = [augment(x, gen) for _ in range(2)]
    ref = sum(F.cross_entropy(a(v), y) + F.cross_entropy(b(v), y) for v in views) / 2
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)


def test_codistill_teacher_is_stop_gradient():
    # Gradient into model b from a's distillation term must be zero: with CE
    # removed (y irrelevant), b's gradient comes only from its own KL term.
    torch.manual_seed(0)
    a, b = SmallCNN().eval(), SmallCNN().eval()
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    full = codistill_loss(a, b, x, y, 1, 1.0, torch.Generator().manual_seed(2))
    gb = torch.autograd.grad(full, list(b.parameters()))
    gen = torch.Generator().manual_seed(2)
    v = augment(x, gen)
    za, zb = a(v), b(v)
    own = F.cross_entropy(zb, y) + F.kl_div(
        F.log_softmax(zb, 1), F.log_softmax(za, 1).detach(), log_target=True,
        reduction="batchmean",
    )  # fmt: skip
    gref = torch.autograd.grad(own, list(b.parameters()))
    for g, r in zip(gb, gref):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-6)


def test_ensemble_averages_member_probabilities():
    torch.manual_seed(0)
    models, x = [SmallCNN(), SmallCNN()], torch.randn(5, 3, 32, 32)
    ref = (predict_proba(models[0], x) + predict_proba(models[1], x)) / 2
    np.testing.assert_allclose(predict_proba_ensemble(models, x), ref, rtol=1e-6)
