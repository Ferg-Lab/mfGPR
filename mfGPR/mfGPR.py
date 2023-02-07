from mfGPR.models import GPRModel, GPRModel_multiFidelity
import matplotlib.pyplot as plt
import networkx as nx
import time


class mfGPR(object):
    """
    A class for Multi-fidelity Gaussian Process Regression (mfGPR).

    This class trains and stores multiple Gaussian Process Regression (GPR) models using data stored in a dictionary. The data
    in the dictionary can be either single-fidelity data or multi-fidelity data (conditioned on other lower-fidelity models).
    The mfGPR class trains the GPR models using `GPRModel` and `GPRModel_multiFidelity` classes from the `mfGPR.models` module.

    Parameters:
        data (dict): A dictionary containing the training data and any additional information, such as
        data standard deviations, conditioning low-fidelity models and multi low-fidelity model weights values.
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
                print(f"Training model '{key}'")
                start = time.time()
                self.train_on_data_dict(value_dict)
                end = time.time()
                print(f"Training model '{key}' completed in {(end - start) / 60} m")

        for key, value_dict in data.items():
            if "condition" in value_dict.keys():
                print(
                    f"Training model '{key}', conditioned on model(s) '{value_dict['condition']}'"
                )
                start = time.time()
                self.train_on_data_dict(value_dict)
                end = time.time()
                print(f"Training model '{key}' completed in {(end - start) / 60} m")

    def train_on_data_dict(self, data_dict):
        if "model" in data_dict.keys():
            return

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

    def _repr_html_(self):
        g = nx.DiGraph()
        g.add_nodes_from([(i, {"name": k}) for i, k in enumerate(self.data.keys())])
        labels = nx.get_node_attributes(g, "name")
        labels_inv = {v: k for k, v in labels.items()}
        e_list = list()
        for k, v in self.data.items():
            if "condition" in v.keys():
                if isinstance(v["condition"], str):
                    e_list.append((labels_inv[v["condition"]], labels_inv[k]))
                else:
                    for cond in v["condition"]:
                        e_list.append((labels_inv[cond], labels_inv[k]))

        g.add_edges_from(e_list)

        ax = plt.gca()
        nx.draw(
            g,
            labels=labels,
            font_weight="bold",
            node_color="green",
            font_size=14,
            pos=nx.shell_layout(g),
            node_size=3000,
            edge_color="red",
            arrowstyle="Fancy, head_length=2, head_width=3",
            width=2,
            ax=ax,
        )
        ax.set_xlim(*[x * 1.2 for x in ax.get_xlim()])
        ax.set_ylim(*[x * 1.2 for x in ax.get_ylim()])
