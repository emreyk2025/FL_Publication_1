import flwr as fl
import lightgbm as lgb
from typing import Dict, List, Tuple

class LGBMClient(fl.client.NumPyClient):
    def __init__(self, train_data, val_data, test_data, num_train, num_val, num_test, model_func, model_params):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.model_func = model_func
        self.model_params = model_params
        self.model = self.model_func(self.model_params)

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[bytes]:
        if getattr(self.model, '_Booster', None) is not None:
            model_bytes = self.model._Booster.model_to_bytes()
            params = [model_bytes]
        else:
            params = [b'']
        return params

    def set_parameters(self, parameters: List[bytes]) -> None:
        if parameters[0]:
            model_bytes = parameters[0]
            booster = lgb.Booster(model_bytes=model_bytes)
            self.model._Booster = booster
        else:
            self.model = self.model_func(self.model_params)
            self.model._Booster = None
    def fit(
        self,
        parameters: List[bytes],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[bytes], int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)

        # Check if the model is fitted
        if getattr(self.model, '_Booster', None) is not None:
            init_booster = self.model._Booster
        else:
            init_booster = None

        self.model.fit(
            self.train_data[0],
            self.train_data[1],
            init_model=init_booster,
            eval_set=[(self.val_data[0], self.val_data[1])],
            eval_metric="mean_squared_error"
        )
        new_parameters = self.get_parameters(config={})
        # Calculate training metrics
        train_preds = self.model.predict(self.train_data[0])
        train_mse = ((self.train_data[1] - train_preds) ** 2).mean()
        return new_parameters, self.num_train, {"train_mse": float(train_mse)}

    def evaluate(
        self,
        parameters: List[bytes],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)
        preds = self.model.predict(self.test_data[0])
        mse = ((self.test_data[1] - preds) ** 2).mean()
        return float(mse), self.num_test, {"MSE": float(mse)}

