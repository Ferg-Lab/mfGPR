import numpy as np
import pytest

from mfGPR.models import GPRModel, GPRModel_multiFidelity


@pytest.fixture
def data():
    np.random.seed(0)
    X = np.random.rand(10, 1)
    Y = np.sin(X).flatten() + np.random.normal(scale=0.1, size=X.shape[0])
    return X, Y[..., None]


@pytest.fixture
def multi_fidelity_data():
    np.random.seed(0)
    X_l = np.random.rand(10, 1)
    X_h = np.random.rand(10, 1)
    Y_l = np.sin(X_l).flatten() + np.random.normal(scale=0.1, size=X_l.shape[0])
    Y_h = np.sin(X_h).flatten() + np.random.normal(scale=0.1, size=X_h.shape[0])
    return X_l, Y_l[..., None], X_h, Y_h[..., None]


def test_GPRModel_predict(data):
    X, Y = data
    model = GPRModel(X, Y, n_samples=2)
    mu, std = model.predict(X)
    assert mu.shape == (X.shape[0], 1)
    assert std.shape == (X.shape[0], 1)


def test_GPRModel_predict_return_samples(data):
    X, Y = data
    model = GPRModel(X, Y, n_samples=2)
    Z = model.predict(X, return_samples=True)
    assert Z.shape == (model.n_samples, X.shape[0])


def test_GPRModel_multiFidelity_predict(multi_fidelity_data):
    X_l, Y_l, X_h, Y_h = multi_fidelity_data
    model_low = GPRModel(X_l, Y_l, n_samples=2)
    model = GPRModel_multiFidelity(X_h, Y_h, model_low, n_samples=2)
    mu, std = model.predict(X_h)
    assert mu.shape == (X_h.shape[0], 1)
    assert std.shape == (X_h.shape[0], 1)


def test_GPRModel_multiFidelity_predict_return_samples(multi_fidelity_data):
    X_l, Y_l, X_h, Y_h = multi_fidelity_data
    model_low = GPRModel(X_l, Y_l, n_samples=2)
    model = GPRModel_multiFidelity(X_h, Y_h, model_low, n_samples=2)
    Z = model.predict(X_h, return_samples=True)
    assert Z.shape == (model.n_samples**2, X_h.shape[0])


def test_GPRModel_multiFidelity_predict_multiCondition(multi_fidelity_data):
    X_l, Y_l, X_h, Y_h = multi_fidelity_data
    model_low = GPRModel(X_l, Y_l, n_samples=2)
    model = GPRModel_multiFidelity(
        X_h, Y_h, [model_low, model_low], theta=[0.5, 0.5], n_samples=2
    )
    mu, std = model.predict(X_h)
    assert mu.shape == (X_h.shape[0], 1)
    assert std.shape == (X_h.shape[0], 1)


def test_GPRModel_multiFidelity_predict_return_samples_multiCondition(
    multi_fidelity_data,
):
    X_l, Y_l, X_h, Y_h = multi_fidelity_data
    model_low = GPRModel(X_l, Y_l, n_samples=2)
    model = GPRModel_multiFidelity(
        X_h, Y_h, [model_low, model_low], theta=[0.5, 0.5], n_samples=2
    )
    Z = model.predict(X_h, return_samples=True)
    assert Z.shape == (model.n_samples**2, X_h.shape[0])
