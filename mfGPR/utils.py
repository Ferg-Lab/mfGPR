import numpy as np


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
