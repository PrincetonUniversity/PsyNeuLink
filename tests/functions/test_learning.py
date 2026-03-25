import numpy as np
import pytest

from psyneulink.library.compositions.autodiffcomposition import torch_available
from psyneulink.core.components.functions.nonstateful.learningfunctions import EMStorage
import psyneulink as pnl

if torch_available:
    import torch

TEST_DATA_STORE = [
    # entries, memory_matrix, storage_location
    (np.array([1., 2.]), np.zeros((2, 2), dtype=float), 1),
]

TEST_DATA_STORE_TORCH = [
    (torch.tensor(el[0].reshape(1, -1), dtype=torch.double), torch.tensor(el[1], dtype=torch.double), el[2])
    for el in TEST_DATA_STORE
]


@pytest.mark.parametrize(
    "entry, memory, storage_location", TEST_DATA_STORE)   
def test_em_storage(entry, memory, storage_location, func_mode):
    f = EMStorage(
        default_variable=[0.0, 0.0],
        axis=0,
        storage_location=storage_location,
        storage_prob=1.,
        seed=0,
    )
    EX = pytest.helpers.get_func_execution(f, func_mode)
    out = EX(entry, memory.copy())

    expected = memory.copy()
    expected[:, storage_location] = entry
    np.testing.assert_array_equal(out, expected)

@pytest.mark.parametrize(
    "entry, memory, storage_location", TEST_DATA_STORE_TORCH)  
def test_em_storage_torch(entry, memory, storage_location):
    """EMStorage writes the entry into the chosen slot of memory_matrix (axis=0: columns)."""
    f = EMStorage(
        default_variable=[0.0, 0.0],
        axis=0,
        storage_location=storage_location,
        storage_prob=1.,
        seed=0,
    )
    context = pnl.Context(execution_id=None)
    pnl_torch_f = f._gen_pytorch_fct("cpu", context=context)
    pnl_out = pnl_torch_f(entry, memory).detach().cpu().numpy()
    expected = memory.detach().cpu().numpy().copy()
    expected[:, storage_location] = entry
    np.testing.assert_array_equal(pnl_out, expected)