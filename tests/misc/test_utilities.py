from collections.abc import Iterable
import numpy as np
import pytest

from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, prune_unused_args


@pytest.mark.parametrize(
    'arr, expected',
    [
        ([[0], [0, 0]], np.array([np.array([0]), np.array([0, 0])], dtype=object)),
        # should test these but numpy cannot easily create an array from them
        # [np.ones((2,2)), np.zeros((2,1))]
        # [np.array([[0]]), np.array([[[ 1.,  1.,  1.], [ 1.,  1.,  1.]]])]

    ]
)
def test_convert_all_elements_to_np_array(arr, expected):
    converted = convert_all_elements_to_np_array(arr)

    # no current numpy methods can test this
    def check_equality_recursive(arr, expected):
        if (
            not isinstance(arr, Iterable)
            or (isinstance(arr, np.ndarray) and arr.ndim == 0)
        ):
            assert arr == expected
        else:
            assert isinstance(expected, type(arr))
            assert len(arr) == len(expected)

            for i in range(len(arr)):
                check_equality_recursive(arr[i], expected[i])

    check_equality_recursive(converted, expected)


def f():
    pass


def g(a):
    pass


def h(a, b=None):
    pass


def i(b=None):
    pass


@pytest.mark.parametrize(
    'func, args, kwargs, expected_pruned_args, expected_pruned_kwargs', [
        (f, 1, {'a': 1}, [], {}),
        (g, 1, {'x': 1}, [1], {}),
        (g, None, {'a': 1}, [], {'a': 1}),
        (h, None, {'a': 1, 'b': 1, 'c': 1}, [], {'a': 1, 'b': 1}),
        (h, [1, 2, 3], None, [1], {}),
        (i, None, {'a': 1, 'b': 1, 'c': 1}, [], {'b': 1}),
        (i, [1, 2, 3], None, [], {}),
    ]
)
def test_prune_unused_args(func, args, kwargs, expected_pruned_args, expected_pruned_kwargs):
    pruned_args, pruned_kwargs = prune_unused_args(func, args, kwargs)

    assert pruned_args == expected_pruned_args
    assert pruned_kwargs == expected_pruned_kwargs
