from flwr.common import Context
from flwr.server import ServerConfig, ServerAppComponents
from utils import evaluate_metrics_aggregation, weighted_average
from flwr.server.strategy import FedXgbBagging, FedAvg



def get_server_fn(fraction_fit, 
                  fraction_evaluate, 
                  min_evaluate_clients, 
                  initial_parameters,
                  model_name,
                  num_clients,
                  num_rounds):
    

    def server_fn(context: Context) -> ServerAppComponents:

        if model_name in ['simple_LSTM', 'stacked_LSTM', 'bidirectional_LSTM', 'GRU']:
                
                strategy = FedAvg(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_evaluate_clients=min_evaluate_clients,
                    min_available_clients=int(num_clients * 0.75),
                    evaluate_metrics_aggregation_fn=weighted_average,
                    initial_parameters=initial_parameters,
                )

                config = ServerConfig(num_rounds=num_rounds)

                return ServerAppComponents(strategy=strategy, config=config)

        
        elif model_name in ['XGBoost', 'LightGBM']:
                if model_name == 'XGBoost':
                        
                        strategy = FedXgbBagging(
                                fraction_fit = fraction_fit,
                                min_fit_clients = int(num_clients * 0.50),
                                min_available_clients = int(num_clients * 0.75),
                                min_evaluate_clients = min_evaluate_clients,
                                fraction_evaluate = fraction_evaluate,
                                evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation,
                        )
                        
                        config = ServerConfig(num_rounds=num_rounds)

                        return ServerAppComponents(strategy=strategy, config=config)


                
                elif model_name == 'LightGBM':
                        
                        strategy = LightGBMStrategy()
                        
                        config = ServerConfig(num_rounds=num_rounds)

                        return ServerAppComponents(strategy=strategy, config=config)
    
    return server_fn

        
