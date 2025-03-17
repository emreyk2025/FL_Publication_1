import flwr
from flwr.server.strategy import Strategy
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional, Dict
import lightgbm as lgb

class LightGBMStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, flwr.common.FitIns]]:
        config = {}
        fit_ins = flwr.common.FitIns(parameters, config)
        # Sample clients for this round
        sample_size = min(1.0, client_manager.num_available())
        clients = client_manager.sample(
            num_clients=int(sample_size)
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, flwr.common.EvaluateIns]]:
        
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}

        # Collect all models from clients
        models = []
        for (_, fit_res) in results:
            params = fit_res.parameters
            model_bytes = params.tensors[0]
            if not model_bytes:
                continue  # Handle empty data
            booster = lgb.Booster(model_bytes=model_bytes)
            models.append((booster, fit_res.num_examples))

        # Custom aggregation logic
        # For example, select the model with the lowest training loss
        best_model = None
        best_loss = float('inf')
        for (model, num_examples), (_, fit_res) in zip(models, results):
            metrics = fit_res.metrics
            train_mse = metrics.get("train_mse", float('inf'))
            if train_mse < best_loss:
                best_loss = train_mse
                best_model = model

        # Serialize the best model
        aggregated_model_bytes = best_model.model_to_bytes()
        aggregated_params = Parameters(
            tensor_type="bytes",
            tensors=[aggregated_model_bytes]
        )

        return aggregated_params, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        return None
