import numpy as np
from itertools import product


def Tchebycheff(x, theta, rho=0.05):
    # doi: 10.1109/TEVC.2005.851274.
    dotted = x * theta.reshape(-1, 1, 1)
    lin = dotted.sum(axis=0)
    non_lin = np.minimum(*dotted)
    return non_lin + rho * lin


def scalarize_linear(x, theta):
    return np.sum(x * theta.reshape(-1, 1, 1), axis=0)


def scalarize_factory(scalarize):
    if scalarize == "ParEGO":
        return Tchebycheff
    if scalarize == "linear":
        return scalarize_linear


def get_scalarization_coefficients(num_terms, discretization):
    """
    Get the coefficients for the scalarization.

    Parameters:
        num_terms (int): Number of terms in the scalarization.
        discretization (int): Number of values to discretize the coefficent values on the interval 0 and 1.

    Returns:
        coefficients (list of lists): A list of lists, where each list represents the coefficients for the scalarization.
    """
    # Generate the discretization values between 0 and 1
    coeffs = np.linspace(0, 1, discretization)

    # Generate all combinations of the discretization values
    combinations = list(product(coeffs, repeat=num_terms - 1))

    # Filter out combinations where the sum is greater than 1
    coefficients = [
        list(np.concatenate(([1 - sum(c)], c))) for c in combinations if sum(c) <= 1
    ]

    return coefficients
