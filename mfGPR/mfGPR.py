from mfGPR.models import GPRModel, GPRModel_multiFidelity


class mfGPR(object):
    """
    A class for Multi-fidelity Gaussian Process Regression (mfGPR).
    
    This class trains and stores multiple Gaussian Process Regression (GPR) models using data stored in a dictionary. The data
    in the dictionary can be either single-fidelity data or multi-fidelity data (conditioned on other lower-fidelity models).
    The mfGPR class trains the GPR models using `GPRModel` and `GPRModel_multiFidelity` classes from the `mfGPR.models` module.
    
    Parameters:
        data (dict): A dictionary containing the training data and any additional information, such as data standard deviations, conditions, or theta values.
        n_samples (int, optional): The number of samples to use in the Monte Carlo approximation. Default is 10.
        
    Attributes:
        data (dict): The input dictionary of training data.
        n_samples (int): The number of samples used in the Monte Carlo approximation.
        
    Example with two fidelity levels:
        ```
        data = {
            "low": {
                "data": [X_train_low, Y_train_low]),
                "std": Y_std_low,
            },
            "high": {
                "data": [X_train, Y_train],
                "condition": "low",
                "std": Y_std_high,
            }
        }
        
        mfGPR_model = mfGPR(data)
        high_pred = mfGPR_model['high'].predict(X_test)
        ```

    Example with two low-fidelity and one high fidelity:
        ```
        data = {
            "low_0": {
                "data": [X_train_low_0, Y_train_low_0]),
            },
            "low_1": {
                "data": [X_train_low_1, Y_train_low_1]),
            },
            "high": {
                "data": [X_train, Y_train],
                "condition": ["low_0", "low_1"],
                "theta": [0.5, 0.5],
            }
        }
        
        mfGPR_model = mfGPR(data)
        high_pred = mfGPR_model['high'].predict(X_test)
        ```

    Example with three fidelity levels:
        ```
        data = {
            "low": {
                "data": [X_train_low, Y_train_low]),
            },
            "mid": {
                "data": [X_train_mid, Y_train_mid]),
                "condition": "low",
            },
            "high": {
                "data": [X_train, Y_train],
                "condition": "mid",
            }
        }
        
        mfGPR_model = mfGPR(data)
        high_pred = mfGPR_model['high'].predict(X_test)
        ```
    """
    def __init__(self, data: dict, n_samples: int = 10):

        self.data = data
        self.n_samples = n_samples

        for key, value_dict in self.data.items():
            if "condition" not in value_dict.keys():
                self.train_on_data_dict(value_dict)

        for key, value in data.items():
            if "condition" in value.keys():
                self.train_on_data_dict(value_dict)

    def train_on_data_dict(self, data_dict):
        if "model" in data_dict.keys():
            pass

        X, Y = data_dict["data"]
        if "std" in data_dict.keys():
            std = data_dict["std"]
        else:
            std = None

        if "condition" not in data_dict.keys():
            data_dict["model"] = GPRModel(X, Y, std=std, n_samples=self.n_samples)
        else:
            condition = data_dict["condition"]
            if isinstance(condition, list):
                assert "theta" in data_dict.keys()
                theta = data_dict["theta"]

                model_lows = list()
                for m in condition:
                    if "model" in self.data[m].keys():
                        model_lows.append(self.data[m]["model"])
                    else:
                        self.train_on_data_dict(self.data[m])
                        model_lows.append(self.data[m]["model"])

                assert len(model_lows) == len(theta)

            elif isinstance(condition, str):
                theta = None
                if "model" in self.data[condition].keys():
                    model_lows = self.data[condition]["model"]
                else:
                    self.train_on_data_dict(self.data[condition])
                    model_lows = self.data[condition]["model"]

            data_dict["model"] = GPRModel_multiFidelity(
                X,
                Y,
                model_lows=model_lows,
                std=std,
                theta=theta,
                n_samples=self.n_samples,
            )

    def __getitem__(self, name):
        assert name in self.data.keys()
        assert "model" in self.data[name].keys()
        return self.data[name]["model"]
