import numpy as np
from mfGPR import mfGPR

X_train_low = np.random.randn(10, 1)
Y_train_low = np.random.randn(10, 1)
Y_std_low = np.random.randn(10)

X_train_mid = np.random.randn(10, 1)
Y_train_mid = np.random.randn(10, 1)

X_train = np.random.randn(10, 1)
Y_train = np.random.randn(10, 1)
Y_std_high = np.random.randn(10)

X_test = np.random.randn(10, 1)


def test_mfGPR_init_with_two_fidelity_levels():
    data = {
        "low": {
            "data": [X_train_low, Y_train_low],
            "std": Y_std_low,
        },
        "high": {
            "data": [X_train, Y_train],
            "condition": "low",
            "std": Y_std_high,
        },
    }

    mfGPR_model = mfGPR(data, n_samples=2)

    mu, std = mfGPR_model["high"].predict(X_test)
    assert mu.shape == (X_test.shape[0], 1)
    assert std.shape == (X_test.shape[0], 1)


def test_mfGPR_init_with_two_low_fidelity_and_one_high_fidelity():
    data = {
        "low_0": {
            "data": [X_train_low, Y_train_low],
        },
        "low_1": {
            "data": [X_train_low, Y_train_low],
        },
        "high": {
            "data": [X_train, Y_train],
            "condition": ["low_0", "low_1"],
            "theta": [0.5, 0.5],
        },
    }

    mfGPR_model = mfGPR(data, n_samples=2)

    mu, std = mfGPR_model["high"].predict(X_test)
    assert mu.shape == (X_test.shape[0], 1)
    assert std.shape == (X_test.shape[0], 1)


def test_mfGPR_init_with_three_fidelity_levels():
    data = {
        "low": {
            "data": [X_train_low, Y_train_low],
        },
        "mid": {
            "data": [X_train_mid, Y_train_mid],
            "condition": "low",
        },
        "high": {
            "data": [X_train, Y_train],
            "condition": "mid",
        },
    }

    mfGPR_model = mfGPR(data, n_samples=2)

    mu, std = mfGPR_model["high"].predict(X_test)
    assert mu.shape == (X_test.shape[0], 1)
    assert std.shape == (X_test.shape[0], 1)


def test_mfGPR_init_with_three_fidelity_levels_mixed():
    data = {
        "low": {
            "data": [X_train_low, Y_train_low],
        },
        "mid": {
            "data": [X_train_mid, Y_train_mid],
            "condition": "low",
        },
        "high": {
            "data": [X_train, Y_train],
            "condition": ["mid", "low"],
            "theta": [0.5, 0.5],
        },
    }

    mfGPR_model = mfGPR(data, n_samples=2)

    mu, std = mfGPR_model["high"].predict(X_test)
    assert mu.shape == (X_test.shape[0], 1)
    assert std.shape == (X_test.shape[0], 1)
