import pytest
import torch

from ca_moe.models import LogisticRegressor, NN1, NN2, LeNet5

BATCH_SIZE = 4
NUM_CLASSES = 10
INPUT_SIZE = 784  # 28 * 28


@pytest.fixture
def x_img() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 1, 28, 28)


@pytest.fixture
def logistic() -> LogisticRegressor:
    return LogisticRegressor(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)


@pytest.fixture
def nn1() -> NN1:
    return NN1(input_size=INPUT_SIZE, hidden_size=128, num_classes=NUM_CLASSES)


@pytest.fixture
def nn2() -> NN2:
    return NN2(input_size=INPUT_SIZE, hidden_sizes=(256, 128), num_classes=NUM_CLASSES)


@pytest.fixture
def lenet5() -> LeNet5:
    return LeNet5(in_channels=1, num_classes=NUM_CLASSES)


@pytest.mark.parametrize("model_fixture", ["logistic", "nn1", "nn2", "lenet5"])
def test_output_shape(model_fixture, x_img, request):
    model = request.getfixturevalue(model_fixture)
    assert model(x_img).shape == (BATCH_SIZE, NUM_CLASSES)


@pytest.mark.parametrize("model_fixture", ["logistic", "nn1", "nn2", "lenet5"])
def test_gradients_flow_to_all_params(model_fixture, x_img, request):
    model = request.getfixturevalue(model_fixture)
    model(x_img).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("model_fixture", ["logistic", "nn1", "nn2", "lenet5"])
def test_eval_mode_is_deterministic(model_fixture, x_img, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x_img), model(x_img))


def test_mlp_param_count_ordering(logistic, nn1, nn2):
    """Param count must increase: logistic < nn1 < nn2 (LeNet5 excluded — CNN)."""
    counts = [sum(p.numel() for p in m.parameters()) for m in (logistic, nn1, nn2)]
    assert counts == sorted(counts)


def test_logistic_no_bias_param_count():
    model = LogisticRegressor(
        input_size=INPUT_SIZE, num_classes=NUM_CLASSES, bias=False
    )
    assert sum(p.numel() for p in model.parameters()) == INPUT_SIZE * NUM_CLASSES
