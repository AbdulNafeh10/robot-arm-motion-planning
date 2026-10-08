"""Matrix and vector operations from the original HW1 exercise."""

import numpy as np


def multiply_matrices(*matrices):
    if not matrices:
        raise ValueError('Provide at least one matrix')
    result = np.asarray(matrices[0])
    for matrix in matrices[1:]:
        result = result @ np.asarray(matrix)
    return result


def multiply_matrix_vector(matrix, vector):
    return np.asarray(matrix) @ np.asarray(vector)


def multiply_transpose_vector(matrix, vector):
    # The vector length must match the rows of the original matrix.
    return np.asarray(matrix).T @ np.asarray(vector)


def solve_matrix_vector(matrix, vector):
    # Solving Ax=p avoids explicitly computing A's inverse.
    return np.linalg.solve(matrix, vector)


def cross_product(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError('Cross product requires two 3D vectors')
    return np.cross(first, second)


def dot_product(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError('Dot product requires equally sized vectors')
    return np.dot(first, second)
