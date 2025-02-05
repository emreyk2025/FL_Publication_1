from logging import INFO
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
import numpy as np
import io
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import xgboost as xgb

num_local_round = 1

# Define Flower Client:

class XgbClient(fl.client.Client):
    def __init__(self, train_Dmatrix, val_Dmatrix, test_Dmatrix, model_func, model_params, num_train, num_val, num_test):
        # Initially we don't have a model in XGBoost
        self.model = None
        self.model_func = model_func
        self.model_params = model_params
        self.config = None                                            
        self.train_Dmatrix = train_Dmatrix
        self.val_Dmatrix = val_Dmatrix
        self.test_Dmatrix = test_Dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="bytes", tensors=[])
        )
    
    # Ins: Instructions, Res: Fit response from client
    def fit(self, ins: FitIns) -> FitRes:
        if not self.model:
            log(INFO, "Start training at round 1")
            model = self.model_func().train(
                self.model_params,
                self.train_Dmatrix,
                num_boost_round = num_local_round,
                evals = [(self.val_Dmatrix, "validate"), (self.train_Dmatrix, "train")]
            )
            self.config = model.save_config()
            self.model = model
        else:
            if ins.parameters.tensors:
                global_model_array = parameters_to_ndarrays(ins.parameters)[0]
                global_model_bytes = global_model_array.tobytes()
                with io.BytesIO() as buffer:
                    buffer.write(global_model_bytes)
                    buffer.seek(0)
                    self.model.load_model(buffer)
                self.model.load_config(self.config)

            model = self._local_boost()
        
        with io.BytesIO() as buffer:
            buffer.write(self.model.save_raw("json"))
            buffer.seek(0)
            model_bytes = buffer.getvalue()

        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_array])
    
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=parameters,
            num_examples=self.num_train,
            metrics={},
        )
    
    def _local_boost(self):
        for i in range (num_local_round):
            self.model.update(self.train_Dmatrix, self.model.num_boosted_rounds())
        
        model = self.model[
            self.model.num_boosted_rounds() - num_local_round : self.model.num_boosted_rounds()
        ]

        return model
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

        if self.model is None:
            if ins.parameters.tensors:
                global_model_array = parameters_to_ndarrays(ins.parameters)[0]
                global_model_bytes = global_model_array.tobytes()
                with io.BytesIO() as buffer:
                    buffer.write(global_model_bytes)
                    buffer.seek(0)
                    self.model = xgb.Booster()
                    self.model.load_model(buffer)
            else:
                raise ValueError("No global model parameters available for evaluation")

        y_pred = self.model.predict(self.test_Dmatrix)
        y_true = self.test_Dmatrix.get_label()

        # Compute Mean Squared Error
        mse = round(((y_true - y_pred) ** 2).mean(), 4)
    
        return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=mse,
                num_examples=self.num_test,         
                metrics={"MSE": mse}            
            )
    
# fl.client.start_client(server_address="192.168.0.104", client=XgbClient())