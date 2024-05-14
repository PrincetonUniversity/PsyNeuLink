from collections.abc import Iterable
import numpy as np
import pytest

from psyneulink.core.globals.utilities import (
    convert_all_elements_to_np_array, extended_array_equal, prune_unused_args, update_array_in_place
)


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


regular_np_array_parametrization = [
    [],
    0,
    np.array([]),
    [0],
    [[[[[0]]]]],
    [[[[[1]]]]],
    [[1], [0]],
    [[[[[0]]], [[[1]]]]],
]


@pytest.mark.parametrize('a', regular_np_array_parametrization)
@pytest.mark.parametrize('b', regular_np_array_parametrization)
def test_extended_array_equal_regular(a, b):
    assert extended_array_equal(a, b) == np.array_equal(a, b)


irregular_np_array_parametrization = [
    ([0, [1, 0]], [0, [1, 0]], True),
    ([1, [1, 0], [[[1]]]], [1, [1, 0], [[[1]]]], True),
    ([np.array([0, 0]), np.array([0])], [np.array([0, 0]), np.array([0])], True),
    ([1, [], [[[]]]], [1, [], [[[]]]], True),
    ([['ab'], None], [['ab'], None], True),
    ([[0, None, 'ab'], [1, 0]], [[0, None, 'ab'], [1, 0]], True),

    ([0, [0, 0]], [0, [1, 0]], False),
    ([1, [1, 0], [[[1]]]], [], False),
    ([1, [], [[[]]]], [], False),
    ([['ab'], None], [['ab'], 0], False),
    ([[0, None, 'a'], [1, 0]], [[0, None, 'ab'], [1, 0]], False),
]


@pytest.mark.parametrize(
    'a, b, equal', irregular_np_array_parametrization
)
def test_extended_array_equal_irregular(a, b, equal):
    assert extended_array_equal(a, b) == equal


@pytest.mark.parametrize(
    'a',
    [x[0] for x in irregular_np_array_parametrization]
    + [x[1] for x in irregular_np_array_parametrization]
)
def test_extended_array_equal_irregular_identical(a):
    assert extended_array_equal(a, a)


@pytest.mark.parametrize(
    'target, source',
    [
        ([[0, 0], [0, 0]], [[1, 1], [1, 1]]),
        ([[0], [0, 0]], [[1], [1, 1]]),
    ]
)
def test_update_array_in_place(target, source):
    target = convert_all_elements_to_np_array(target)
    source = convert_all_elements_to_np_array(source)
    old_target = target

    update_array_in_place(target, source)

    len_target = len(target)
    assert len_target == len(source)
    assert len_target == len(old_target)
    for i in range(len_target):
        np.testing.assert_array_equal(target[i], source[i])
        np.testing.assert_array_equal(old_target[i], source[i])
        np.testing.assert_array_equal(target[i], old_target[i])


@pytest.mark.parametrize(
    'target, source',
    [
        ([[0], [0, 0]], [[1], [1, 1, 1]]),
        ([0, [0, 0]], [[1], [1, 1]]),
    ]
)
def test_update_array_in_place_failures(target, source):
    target = convert_all_elements_to_np_array(target)
    source = convert_all_elements_to_np_array(source)
    old_target = target

    with pytest.raises(ValueError):
        update_array_in_place(target, source)

    len_target = len(target)
    assert len_target == len(source)
    assert len_target == len(old_target)
    for i in range(len_target):
        assert not np.array_equal(target[i], source[i])
        assert not np.array_equal(old_target[i], source[i])
        np.testing.assert_array_equal(target[i], old_target[i])
