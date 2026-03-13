import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ca_moe.models import CostAwareMoE, LogisticRegressor, NN1, NN2, LeNet5
from ca_moe.factories import get_cost_fn

BATCH_SIZE = 8
NUM_CLASSES = 10
NUM_EXPERTS = 4
INPUT_SHAPE = (1, 28, 28)  # C, H, W


@pytest.fixture
def experts() -> dict:
    return {
        "logistic": LogisticRegressor(input_size=784, num_classes=NUM_CLASSES),
        "nn1": NN1(input_size=784, hidden_size=128, num_classes=NUM_CLASSES),
        "nn2": NN2(input_size=784, hidden_sizes=(256, 128), num_classes=NUM_CLASSES),
        "lenet5": LeNet5(in_channels=1, num_classes=NUM_CLASSES),
    }


@pytest.fixture
def gating() -> LogisticRegressor:
    return LogisticRegressor(input_size=784, num_classes=NUM_EXPERTS)


@pytest.fixture
def cost_fn():
    return get_cost_fn("logflops", input_shape=(1, *INPUT_SHAPE))


@pytest.fixture
def moe(experts, gating, cost_fn) -> CostAwareMoE:
    model = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.1,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="adam",
    )
    model.log = lambda *args, **kwargs: None
    return model


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, *INPUT_SHAPE)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return x, y


def test_experts_in_train_mode_after_init(moe):
    """eval_mode used during __init__ (FLOPs measurement) must be fully released."""
    for name, expert in moe.experts.items():
        assert expert.training, f"Expert '{name}' is in eval mode after __init__"


def test_costs_are_not_parameters(moe):
    """Buffers must not appear in parameters() - no optimizer update"""
    param_names = {n for n, _ in moe.named_parameters()}
    assert "costs" not in param_names
    assert "raw_costs" not in param_names


def test_forward_probas_sum_to_one(moe, batch):
    x, _ = batch
    probas, _ = moe(x)
    assert torch.allclose(probas.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def test_gamma_zero_loss_equals_soft_ce(experts, gating, cost_fn, batch):
    """When gamma=0 and lambda_=0, loss == E_sample[sum_j p_j * CE_j]."""
    moe = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.0,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="adam",
    )
    x, y = batch
    probas, outputs = moe(x)
    expected = (
        probas.T
        * torch.stack([F.cross_entropy(out, y, reduction="none") for out in outputs])
    ).sum() / y.size(0)
    actual, _ = moe._compute_loss(probas, outputs, y)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_lambda_modifies_loss_by_entropy(experts, gating, cost_fn, batch):
    """loss(lambda=L) == loss(lambda=0) - L * entropy, verified on the same forward pass."""
    L = 0.3
    moe = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.0,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="adam",
    )
    x, y = batch
    probas, outputs = moe(x)
    loss_0, metrics = moe._compute_loss(probas, outputs, y)
    loss_L, _ = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.0,
        lambda_=L,
        learning_rate=1e-3,
        optimizer="adam",
    )._compute_loss(probas, outputs, y)
    assert torch.allclose(loss_L, loss_0 - L * metrics.gating_entropy, atol=1e-6)


def test_costs_frozen_after_backward(moe, batch):
    """Optimizer step must not modify cost buffers."""
    costs_before = moe.costs.clone()
    raw_costs_before = moe.raw_costs.clone()
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)
    optimizer.zero_grad()
    moe.training_step(batch, batch_idx=0).backward()
    optimizer.step()
    assert torch.equal(moe.costs, costs_before)
    assert torch.equal(moe.raw_costs, raw_costs_before)


def test_gating_params_update_after_backward(moe, batch):
    """Gating parameters must receive gradient updates."""
    params_before = {k: v.clone() for k, v in moe.gating.named_parameters()}
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.5)
    optimizer.zero_grad()
    moe.training_step(batch, batch_idx=0).backward()
    optimizer.step()
    assert any(
        not torch.equal(v, params_before[k]) for k, v in moe.gating.named_parameters()
    )


def test_expert_utilization_sums_to_100(moe):
    x = torch.randn(64, *INPUT_SHAPE)
    y = torch.randint(0, NUM_CLASSES, (64,))
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    util = moe.get_expert_utilization(dl)
    assert abs(sum(util.values()) - 100.0) < 1e-3


def test_configure_optimizers_unknown_raises(experts, gating, cost_fn):
    moe = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.0,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="rmsprop",
    )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        moe.configure_optimizers()
