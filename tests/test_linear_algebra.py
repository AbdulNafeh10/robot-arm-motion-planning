import numpy as np
import pytest

from src.linear_algebra import (
    multiply_matrices, multiply_matrix_vector, multiply_transpose_vector,
    solve_matrix_vector, cross_product, dot_product,
)


@pytest.mark.parametrize('a,b,expected', [
    ([[4, 2], [21, 2]], [[1, 23], [4, 12]], [[12, 116], [29, 507]]),
    ([[4, 27], [43, 45]], [[75, 64], [17, 82]], [[759, 2470], [3990, 6442]]),
])
def test_matrix_product_from_hw1(a, b, expected):
    np.testing.assert_array_equal(multiply_matrices(a, b), expected)


@pytest.mark.parametrize('a,p,expected', [
    ([[4, 2], [21, 2]], [1, 2], [8, 25]),
    ([[4, 27], [43, 45]], [2, 3], [89, 221]),
])
def test_matrix_vector_from_hw1(a, p, expected):
    np.testing.assert_array_equal(multiply_matrix_vector(a, p), expected)


@pytest.mark.parametrize('a,p,expected', [
    ([[4, 2], [21, 2]], [1, 2], [46, 6]),
    ([[4, 27], [43, 45]], [2, 3], [137, 189]),
])
def test_transpose_vector_from_hw1(a, p, expected):
    np.testing.assert_array_equal(multiply_transpose_vector(a, p), expected)


def test_transpose_rectangular_regression():
    # The original implementation rejected this valid 3x2 example.
    a = [[1, 2], [3, 4], [5, 6]]
    np.testing.assert_array_equal(multiply_transpose_vector(a, [7, 8, 9]),
                                  [76, 100])


@pytest.mark.parametrize('a,p,expected', [
    ([[4, 2], [21, 2]], [1, 2], [1/17, 13/34]),
    ([[4, 27], [43, 45]], [2, 3], [-9/981, 74/981]),
])
def test_solve_from_hw1(a, p, expected):
    np.testing.assert_allclose(solve_matrix_vector(a, p), expected)


@pytest.mark.parametrize('a,b,expected', [
    ([12, 30, 20], [30, 1, 1], [10, 588, -888]),
    ([3, -3, 1], [4, 92, 2], [-98, -2, 288]),
])
def test_cross_from_hw1(a, b, expected):
    np.testing.assert_array_equal(cross_product(a, b), expected)


@pytest.mark.parametrize('a,b,expected', [
    ([16, 25], [12, 35], 1067),
    ([25, 56, 17], [58, 9, 10], 2124),
])
def test_dot_from_hw1(a, b, expected):
    assert dot_product(a, b) == expected


def test_chain_from_hw1():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    np.testing.assert_array_equal(multiply_matrices(a, b, np.eye(2)), [[19, 22], [43, 50]])


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        multiply_matrices()
