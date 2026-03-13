import torch.nn as nn

from ca_moe.utils.tools import eval_mode


def _simple_module(training: bool) -> nn.Module:
    m = nn.Linear(4, 2)
    m.train(training)
    return m


def test_eval_mode_sets_eval():
    m = _simple_module(training=True)
    with eval_mode(m):
        assert not m.training


def test_eval_mode_restores_train():
    m = _simple_module(training=True)
    with eval_mode(m):
        pass
    assert m.training

# should remain eval after context exits
def test_eval_mode_preserves_eval():
    m = _simple_module(training=False)
    with eval_mode(m):
        assert not m.training
    assert not m.training


def test_eval_mode_restores_on_exception():
    m = _simple_module(training=True)
    try:
        with eval_mode(m):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.training
