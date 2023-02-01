import GPy
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (
    make_non_linear_kernels,
)
import numpy as np
from mfGPR.utils import scalarize_factory


class GPRModel(object):
    def __init__(self, X, Y, std=None, base_kernel=GPy.kern.RBF, n_samples: int = 10):
        self.X = X
        self.Y = Y
        if std is None:
            self.std = np.random.normal(scale=0.1, size=self.X.shape[0])
        else:
            self.std = std
        kernel = make_non_linear_kernels(
            base_kernel, n_fidelities=1, n_input_dims=X.shape[1]
        )[0]
        self.kernel = kernel
        self.n_samples = n_samples

        self.fit()

    def fit(self):
        var = np.diag(self.std.flatten() ** 2)

        k_low = self.kernel
        noise = GPy.kern.Fixed(1, var)
        noise.fix()
        k = k_low + noise
        self.model = GPy.models.GPRegression(self.X, self.Y, k)
        self.model.Gaussian_noise.fix(1e-5)
        self.model.optimize_restarts(verbose=False, robust=True, num_restarts=100)
        self.kernel_predict = self.model.kern.kern_fidelity_1.copy()
        return self

    def predict(self, X, return_samples: bool = False):
        if return_samples is False:
            mu, var = self.model.predict(X, kern=self.kernel_predict)
            return mu, np.sqrt(var)
        else:
            mu, C = self.model.predict(X, kern=self.kernel_predict, full_cov=True)
            Z = np.random.multivariate_normal(mu.flatten(), C, self.n_samples)
            return Z


class GPRModel_multiFidelity(object):
    def __init__(
        self,
        X,
        Y,
        model_lows,
        std=None,
        base_kernel=GPy.kern.RBF,
        scalarize="linear",
        theta=None,
        n_samples=10,
    ):
        self.X = X
        self.Y = Y

        if std is None:
            self.std = np.random.normal(scale=0.1, size=self.X.shape[0])
        else:
            self.std = std

        kernel = make_non_linear_kernels(
            base_kernel, n_fidelities=2, n_input_dims=X.shape[1]
        )[1]
        self.kernel = kernel

        self.model_lows = model_lows
        self.n_samples = n_samples

        if isinstance(self.model_lows, list):
            self.isMultiFidelity = len(self.model_lows)
        elif isinstance(self.model_lows, GPRModel) or isinstance(
            self.model_lows, GPRModel_multiFidelity
        ):
            self.isMultiFidelity = False
        else:
            raise ValueError()

        self.theta = theta
        if self.theta is None:
            assert self.isMultiFidelity == False
        elif isinstance(self.theta, list):
            assert len(self.theta) == self.isMultiFidelity
            self.theta = np.array(self.theta)
        elif isinstance(self.theta, np.ndarray):
            assert len(self.theta) == self.isMultiFidelity
        else:
            raise ValueError()

        self.scalarize = scalarize_factory(scalarize)

        self.fit()

    def get_mean_low(self, X, return_samples: bool = False):
        if self.isMultiFidelity is False:
            if return_samples is False:
                mean_low, _ = self.model_lows.predict(X)
            else:
                mean_low = self.model_lows.predict(X, return_samples=True)
        else:
            if return_samples is False:
                means_low = np.concatenate(
                    [m.predict(X)[0].reshape(1, -1, 1) for m in self.model_lows], axis=0
                )
                mean_low = self.scalarize(means_low, self.theta)
            else:
                means_low = np.concatenate(
                    [
                        m.predict(X, return_samples=True).T[None]
                        for m in self.model_lows
                    ],
                    axis=0,
                )
                mean_low = self.scalarize(means_low, self.theta).T

        return mean_low

    def fit(self):
        var = np.diag(self.std.flatten() ** 2)

        mean_low = self.get_mean_low(self.X)
        augmented_input = np.concatenate([self.X, mean_low], axis=1)

        k_high = self.kernel
        noise = GPy.kern.Fixed(1, var)
        noise.fix()
        k = k_high + noise
        self.model = GPy.models.GPRegression(augmented_input, self.Y, k)
        self.model.Gaussian_noise.fix(1e-5)

        self.model.sum.mul.previous_fidelity_fidelity2.variance.fix(1.0)

        self.model.optimize_restarts(verbose=False, robust=True, num_restarts=100)

        self.kernel_predict = (
            self.model.kern.mul.copy() + self.model.kern.bias_kernel_fidelity2.copy()
        )

        return self

    def predict(self, X, return_samples=False):

        mean_lows = self.get_mean_low(X, return_samples=True)

        if return_samples:
            Zs = list()
        else:
            tmp_m = np.empty((mean_lows.shape[0], X.shape[0]))
            tmp_v = np.empty((mean_lows.shape[0], X.shape[0]))

        for i in range(mean_lows.shape[0]):
            augmented_input = np.concatenate([X, mean_lows[i, :][:, None]], axis=1)

            if return_samples:
                mu, C = self.model.predict(
                    augmented_input, kern=self.kernel_predict, full_cov=True
                )
                Z = np.random.multivariate_normal(mu.flatten(), C, self.n_samples)
                Zs.append(Z)
            else:
                mu, v = self.model.predict(augmented_input, kern=self.kernel_predict)

                tmp_m[i, :] = mu.flatten()
                tmp_v[i, :] = v.flatten()

        # get posterior mean and variance
        if return_samples:
            return np.concatenate(Zs, 0)
        else:
            mean = np.mean(tmp_m, axis=0)[:, None]
            var = np.mean(tmp_v, axis=0)[:, None] + np.var(tmp_m, axis=0)[:, None]
            var = np.abs(var)
            std = np.sqrt(var)
            return mean, std
