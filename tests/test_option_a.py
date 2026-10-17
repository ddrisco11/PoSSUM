import numpy as np
import torch

from possum.option_a import router_weights


def test_self_router_preserves_probability_mass_and_cap():
    model = torch.nn.Linear(384, 5)
    with torch.no_grad():
        model.bias[-1] = 100
        weights = router_weights(model, torch.randn(11, 384), 'router_self').numpy()
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
    assert (weights >= 0).all()
    assert (weights[:, -1] <= .25).all()
    component_probabilities = np.random.default_rng(42).random(weights.shape)
    combined = (weights * component_probabilities).sum(1)
    assert (combined >= component_probabilities.min(1)).all()
    assert (combined <= component_probabilities.max(1)).all()


def test_router_without_self_normalizes_extreme_logits():
    model = torch.nn.Linear(384, 4)
    with torch.no_grad():
        model.bias[:] = torch.tensor([-1000, 1000, 0, -500])
        weights = router_weights(model, torch.zeros(2, 384), 'router').numpy()
    assert np.isfinite(weights).all()
    np.testing.assert_array_equal(weights[:, 1], 1)
    np.testing.assert_allclose(weights.sum(1), 1)
