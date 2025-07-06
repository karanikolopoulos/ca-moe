import numpy as np
import torch
import pytest
import tensorflow as tf

from tensorflow.keras.ops import convert_to_tensor as to_tensor
from ca_moe.model.regularizers import L1, L2, L1L2, Orthogonal


@pytest.fixture(scope="module")
def test_data() -> list[np.ndarray]:
    np.random.seed(2025)
    torch.manual_seed(2025)
    tf.random.set_seed(2025)
    
    return [
        np.random.random((4, 4)).astype(np.float32),
        np.zeros((4, 4), dtype=np.float32),
        np.array([[5.0]], dtype=np.float32),
        np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32),
        np.array([[1e-4, 2e-5]], dtype=np.float32)
    ]


@pytest.mark.parametrize("l1", [0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("data_idx", range(5))
def test_l1_equivalence(l1: float, data_idx: int, test_data: list[np.ndarray]) -> None:
    data = test_data[data_idx]
    keras_value = tf.keras.regularizers.L1(l1)(to_tensor(data)).numpy().item()
    torch_value = L1(l1)(torch.tensor(data)).detach().item()
    assert pytest.approx(keras_value, rel=1e-5) == torch_value


@pytest.mark.parametrize("l2", [0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("data_idx", range(5))
def test_l2_equivalence(l2: float, data_idx: int, test_data: list[np.ndarray]) -> None:
    data = test_data[data_idx]
    keras_value = tf.keras.regularizers.L2(l2)(to_tensor(data)).numpy().item()
    torch_value = L2(l2)(torch.tensor(data)).detach().item()
    assert pytest.approx(keras_value, rel=1e-5) == torch_value


@pytest.mark.parametrize("l1, l2", [
    (0.1, 0.2),
    (0.0, 0.1),
    (0.1, 0.0),
    (0.01, 0.01),
    (0.5, 0.5)
])
@pytest.mark.parametrize("data_idx", range(5))
def test_l1l2_equivalence(
    l1: float,
    l2: float,
    data_idx: int,
    test_data: list[np.ndarray]
) -> None:
    data = test_data[data_idx]
    keras_value = tf.keras.regularizers.L1L2(l1=l1, l2=l2)(to_tensor(data)).numpy().item()
    torch_value = L1L2(l1=l1, l2=l2)(torch.tensor(data)).detach().item()
    assert pytest.approx(keras_value, rel=1e-5) == torch_value


@pytest.mark.parametrize("factor, mode", [
    (0.01, "rows"),
    (0.1, "rows"),
    (0.01, "columns"),
    (0.1, "columns")
])
@pytest.mark.parametrize("matrix", [
    np.random.random((5, 5)).astype(np.float32),
    np.random.random((100, 100)).astype(np.float32),
    np.random.random((1000, 1000)).astype(np.float32),
])
def test_orthogonal_equivalence(factor: float, mode: str, matrix: np.ndarray) -> None:
    keras_value = tf.keras.regularizers.OrthogonalRegularizer(factor=factor, mode=mode)(to_tensor(matrix)).numpy().item()
    torch_value = Orthogonal(factor=factor, mode=mode)(torch.tensor(matrix)).detach().item()
    assert pytest.approx(keras_value, rel=1e-3) == torch_value


def test_l1_gradient() -> None:
    data: torch.Tensor = torch.tensor(np.random.randn(5, 5).astype(np.float32), requires_grad=True)
    reg = L1(l1=0.1)
    loss = reg(data)
    loss.backward()
    assert torch.all(data.grad.abs() <= 0.1 + 1e-6)


def test_l2_gradient() -> None:
    data: torch.Tensor = torch.tensor(np.random.randn(5, 5).astype(np.float32), requires_grad=True)
    reg = L2(l2=0.1)
    loss = reg(data)
    loss.backward()
    expected_grad = 2 * 0.1 * data
    assert torch.allclose(data.grad, expected_grad, atol=1e-6)


def test_l1l2_gradient() -> None:
    data: torch.Tensor = torch.tensor(np.random.randn(5, 5).astype(np.float32), requires_grad=True)
    reg = L1L2(l1=0.1, l2=0.1)
    loss = reg(data)
    loss.backward()
    expected_grad = torch.sign(data) * 0.1 + 2 * 0.1 * data
    assert torch.allclose(data.grad, expected_grad, atol=1e-6)


@pytest.mark.parametrize("mode", ["rows", "columns"])
def test_orthogonal_gradient(mode: str) -> None:
    data: torch.Tensor = torch.tensor(np.random.randn(5, 5).astype(np.float32), requires_grad=True)
    reg = Orthogonal(factor=0.1, mode=mode)
    loss = reg(data)
    loss.backward()
    assert data.grad is not None
    assert data.grad.shape == data.shape
    assert torch.all(torch.isfinite(data.grad))
    assert torch.any(data.grad.abs() > 1e-6)


def test_orthogonal_invalid_shape() -> None:
    reg = Orthogonal(factor=0.1, mode="rows")
    with pytest.raises(AssertionError):
        reg(torch.tensor([1.0, 2.0, 3.0]))


def test_orthogonal_invalid_mode() -> None:
    with pytest.raises(AssertionError):
        Orthogonal(factor=0.1, mode="invalid")


def test_negative_regularization_parameters() -> None:
    with pytest.raises(ValueError):
        L1(l1=-0.1)

    with pytest.raises(ValueError):
        L2(l2=-0.1)