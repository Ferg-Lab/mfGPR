import GPy
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (
    make_non_linear_kernels,
)
import numpy as np
from mfGPR.utils import scalarize_factory


class GPRModel(object):
    """A Gaussian Process Regression Model.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, n_features)
        Input features for training.
    Y : np.ndarray, shape=(n_samples, 1)
        Output values for training.
    std : np.ndarray, shape=(n_samples,), optional
        Per-point standard deviation for each sample. If None, it will be set to zeros
        and instead a noise parameter will be learned. Default is None.
    base_kernel : GPy.kern, optional
        The base kernel for the Gaussian Process. Default is GPy.kern.RBF.
    n_samples : int, optional
        The number of samples to draw from the Gaussian Process for the
        purpose of estimating the posterior distribution. Default is 10.
    """

    def __init__(self, X, Y, std=None, base_kernel=GPy.kern.RBF, n_samples: int = 10):
        self.X = X
        self.Y = Y
        if std is None:
            # self.std = np.random.normal(scale=0.1, size=self.X.shape[0])
            self.std = np.zeros(self.X.shape[0])
            self.fixNoise = False
        else:
            self.std = std
            self.fixNoise = True

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

        if self.fixNoise:
            self.model.Gaussian_noise.fix(1e-5)

        self.model.optimize_restarts(verbose=False, robust=True, num_restarts=100)
        self.kernel_predict = self.model.kern.kern_fidelity_1.copy()
        return self

    def predict(self, X, return_samples: bool = False):
        """Make predictions using the Gaussian Process Regression Model.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_features)
            Input features for prediction.
        return_samples : bool, optional
            Whether to return samples from the Gaussian Process. Default is False.

        Returns
        -------
        if return_samples is False:
            mu : np.ndarray, shape=(n_samples, 1)
                The mean prediction.
            std : np.ndarray, shape=(n_samples, 1)
                The standard deviation of the prediction.
        if return_samples is True:
            Z : np.ndarray, shape=(n_samples, self.n_samples)
                Samples from the Gaussian Process.
        """
        if return_samples is False:
            mu, var = self.model.predict(X, kern=self.kernel_predict)
            return mu, np.sqrt(var)
        else:
            mu, C = self.model.predict(X, kern=self.kernel_predict, full_cov=True)
            Z = np.random.multivariate_normal(mu.flatten(), C, self.n_samples)
            return Z


class GPRModel_multiFidelity(object):
    """
    Gaussian Process Regression Model for multi-fidelity data.

        Parameters
    ----------
    X: numpy.ndarray
        Input data with shape (n_samples, n_input_dims).
    Y: numpy.ndarray
        Output data with shape (n_samples, 1).
    model_lows: list, GPRModel, or GPRModel_multiFidelity
        List of low-fidelity GPR models, a single low-fidelity GPR model, or a multi-fidelity GPR model.
    std: numpy.ndarray, optional
        Standard deviation of the noise, by default None.
    base_kernel: GPy.kern, optional
        Base kernel for the model, by default GPy.kern.RBF.
    scalarize: str, optional
        Scalarizing function, by default 'linear'.
    theta: numpy.ndarray, optional
        Coefficients for scalarizing function, by default None.
    n_samples: int, optional
        Number of samples to draw for prediction, by default 10.


    Methods
    -------
    get_mean_low(X, return_samples=False)
        Returns the mean of low-fidelity outputs for the given input X.
    fit()
        Fits the Gaussian process regression model.
    predict(X, return_samples=False)
        Predicts the output for the given input X.
    """
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
            # self.std = np.random.normal(scale=0.1, size=self.X.shape[0])
            self.std = np.zeros(self.X.shape[0])
            self.fixNoise = False
        else:
            self.std = std
            self.fixNoise = True

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
            assert self.isMultiFidelity is False
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
        """Get the mean or samples of the low-fidelity model.
    
        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_features)
            Input features for prediction.
        return_samples : bool, optional
            Whether to return samples from the Gaussian Process. Default is False.
        
        Returns
        -------
        if return_samples is False:
            mu : np.ndarray, shape=(n_samples, 1)
                The mean prediction of the low-fidelity model.
            std : np.ndarray, shape=(n_samples, 1)
                The standard deviation of the prediction of the low-fidelity model.
        if return_samples is True:
            Z : np.ndarray, shape=(n_samples, self.n_samples)
                Samples from the low-fidelity model.
        """
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

        if self.fixNoise:
            self.model.Gaussian_noise.fix(1e-5)

        self.model.sum.mul.previous_fidelity_fidelity2.variance.fix(1.0)

        self.model.optimize_restarts(verbose=False, robust=True, num_restarts=100)

        self.kernel_predict = (
            self.model.kern.mul.copy() + self.model.kern.bias_kernel_fidelity2.copy()
        )

        return self

    def predict(self, X, return_samples=False):
        """Make predictions using the Gaussian Process Regression Model.
        
        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_features)
            Input features for prediction.
        return_samples : bool, optional
            Whether to return samples from the Gaussian Process. Default is False.
        
        Returns
        -------
        if return_samples is False:
            mu : np.ndarray, shape=(n_samples, 1)
                The mean prediction.
            std : np.ndarray, shape=(n_samples, 1)
                The standard deviation of the prediction.
        if return_samples is True:
            Z : np.ndarray, shape=(n_samples, self.n_samples)
                Samples from the Gaussian Process.
        """

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
