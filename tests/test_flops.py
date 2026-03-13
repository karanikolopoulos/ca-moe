import pytest
import torch

from ca_moe.models import CostAwareMoE, LogisticRegressor, NN1, NN2, LeNet5
from ca_moe.factories import get_cost_fn

INPUT_SHAPE = (1, 1, 28, 28)
NUM_CLASSES = 10
EXPERT_ORDER = ["logistic", "nn1", "nn2", "lenet5"]


@pytest.fixture(scope="module")
def experts() -> dict:
    return {
        "logistic": LogisticRegressor(input_size=784, num_classes=NUM_CLASSES),
        "nn1": NN1(input_size=784, hidden_size=128, num_classes=NUM_CLASSES),
        "nn2": NN2(input_size=784, hidden_sizes=(256, 128), num_classes=NUM_CLASSES),
        "lenet5": LeNet5(in_channels=1, num_classes=NUM_CLASSES),
    }


@pytest.fixture(scope="module")
def moe(experts) -> CostAwareMoE:
    gating = LogisticRegressor(input_size=784, num_classes=len(experts))
    return CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=get_cost_fn("logflops", input_shape=INPUT_SHAPE),
        gamma=0.1,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="adam",
    )


def test_costs_sorted_ascending(moe):
    """log10(FLOPs) must increase: logistic < nn1 < nn2 < lenet5."""
    costs = moe.costs.tolist()
    assert costs == sorted(costs), (
        f"Expected ascending costs, got {dict(zip(EXPERT_ORDER, costs))}"
    )


def test_costs_are_positive_and_finite(moe):
    assert (moe.costs > 0).all()
    assert torch.isfinite(moe.costs).all()


def test_lenet5_logistic_flop_gap_exceeds_one_order_of_magnitude(moe):
    """CNN vs linear layer must differ by >10× in FLOPs (log gap > 1)."""
    lr_cost, *_, lenet_cost = moe.costs.tolist()
    assert lenet_cost - lr_cost > 1.0


@pytest.mark.parametrize("cost_name", ["logflops", "logparams", "numlayers"])
def test_cost_ordering_mlps(experts, cost_name):
    """All cost functions must agree: logistic < nn1 < nn2 (LeNet5 excluded — CNN)."""
    kwargs = {"input_shape": INPUT_SHAPE} if cost_name == "logflops" else {}
    cost_fn = get_cost_fn(cost_name, **kwargs)
    gating = LogisticRegressor(input_size=784, num_classes=len(experts))
    m = CostAwareMoE(
        experts=experts,
        gating=gating,
        cost_fn=cost_fn,
        gamma=0.0,
        lambda_=0.0,
        learning_rate=1e-3,
        optimizer="adam",
    )
    logistic_c, nn1_c, nn2_c, *_ = m.costs.tolist()
    assert logistic_c < nn1_c < nn2_c, (
        f"{cost_name}: expected logistic < nn1 < nn2, got {[logistic_c, nn1_c, nn2_c]}"
    )
