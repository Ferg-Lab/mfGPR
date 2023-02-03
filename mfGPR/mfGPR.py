from mfGPR.models import GPRModel, GPRModel_multiFidelity


class mfGPR(object):
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
