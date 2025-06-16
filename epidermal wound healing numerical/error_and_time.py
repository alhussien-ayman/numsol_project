
import numpy as np
from copy import deepcopy
import json

from explicit_finite_difference import solve_finite_difference_explicit
from implicit import solve_finite_difference_implicit
from crank_nicolson import solve_crank_nicolson_newton
from finite_element_method import solve_finite_element_method
from method_of_lines import solve_method_of_lines
from analytical import solve_analytical

method_labels = {
    'solve_finite_difference_explicit': 'Explicit FD',
    'solve_finite_difference_implicit': 'Implicit FD',
    'solve_crank_nicolson_newton': 'Crank-Nicolson (Newton)',
    'solve_finite_element_method': 'Finite Element Method',
    'solve_method_of_lines': 'Method of Lines',
    'solve_analytical': 'Analytical Solution'
}

def compute_relative_error(params, method1, method2):
    name1 = method1.__name__
    name2 = method2.__name__
    label1 = method_labels.get(name1, name1)
    label2 = method_labels.get(name2, name2)

    result1 = method1(deepcopy(params))
    result2 = method2(deepcopy(params))

    sol1 = np.array(result1['result']['solutions'])  # Shape: (num_time_steps, num_space_points)
    sol2 = np.array(result2['result']['solutions'])
    time_labels = result1['tOutputLabels']

    assert sol1.shape == sol2.shape, "Shape mismatch between methods' solutions."

    relative_errors = []
    for u1, u2 in zip(sol1, sol2):
        norm_ref = np.linalg.norm(u1)
        if norm_ref == 0:
            err = 0.0
        else:
            err = np.linalg.norm(u2 - u1) / norm_ref
        relative_errors.append(err)

    return {
        "relative_errors": relative_errors,
        "time_labels": time_labels,
        "label1": label1,
        "label2": label2,
        "time_1": result1['result']['time'],
        "time_2": result2['result']['time'],
        "matrix_1": sol1.tolist(),  # Full matrix for method1
        "matrix_2": sol2.tolist(),  # Full matrix for method2
    }
