import flwr as fl
from tensorflow import convert_to_tensor
from Flower_Client import FlowerClient
#from Fl_Client_XGB import XgbClient
#from FL_Client_LGBM import LGBMClient
from utils import create_sequences
import xgboost as xgb
from flwr.client import ClientApp, Client
from flwr.client.mod import LocalDpMod
from flwr.common import Context



# Construct a FlowerClient with its own data set partition.

def get_client_fn(splitted_data, model_name, model_func, target_column, differential_privacy, model_params):
    '''
    
    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.

    '''

    def client_fn(context: Context) -> Client:
        
        
        cid = int(context.node_config["partition-id"])
                                                    
        
        # Extract partition for client with id = cid
        trainset = splitted_data["train"][cid]
        valset = splitted_data["validation"][cid]
        testset = splitted_data["test"][cid]


        # Prepare data
        x_train = trainset.drop(columns=[target_column])
        x_val = valset.drop(columns=[target_column])
        y_val = valset[target_column]
        x_test = testset.drop(columns=[target_column])
        y_test = testset[target_column]
        

        # Why converting to tensors: compatible with many machine learning models and efficiency

        if model_name in ['simple_LSTM', 'stacked_LSTM', 'bidirectional_LSTM', 'GRU']:

            # Prepare sequences
            time_steps = 30
            x_train_seq, y_train_seq = create_sequences(x_train, y_train, time_steps)
            x_val_seq, y_val_seq = create_sequences(x_val, y_val, time_steps)
            x_test_seq, y_test_seq = create_sequences(x_test, y_test, time_steps)
            
            num_features = x_train_seq.shape[2]
            model = model_func(time_steps, num_features, **model_params)

            # Convert data to tensors
            x_train_tensor = convert_to_tensor(x_train_seq)
            y_train_tensor = convert_to_tensor(y_train_seq)
            x_val_tensor = convert_to_tensor(x_val_seq)
            y_val_tensor = convert_to_tensor(y_val_seq)
            x_test_tensor = convert_to_tensor(x_test_seq)
            y_test_tensor = convert_to_tensor(y_test_seq)

            # Create LSTM/GRU client
            client_instance = FlowerClient(
                x_train=x_train_tensor,
                y_train=y_train_tensor,
                x_val=x_val_tensor,
                y_val=y_val_tensor,
                x_test=x_test_tensor,
                y_test=y_test_tensor,
                model_func=model,
            ).to_client()

            return client_instance
        
        elif model_name in ['XGBoost', 'LightGBM']:
            
            if model_name == 'XGBoost':

                # XGBoost-specific preparation
                train_DMatrix = xgb.DMatrix(x_train, label=y_train)
                val_DMatrix = xgb.DMatrix(x_val, label=y_val)
                test_DMatrix = xgb.DMatrix(x_test, label=y_test)

                client_instance = XgbClient(
                    train_Dmatrix=train_DMatrix,
                    val_Dmatrix=val_DMatrix,
                    test_Dmatrix=test_DMatrix,
                    model_func=model_func,
                    model_params = model_params,
                    num_train=len(x_train),
                    num_val=len(x_val),
                    num_test=len(x_test),
                )

                return client_instance
            
            elif model_name == 'LightGBM':

                train_data = (x_train, y_train)
                val_data = (x_val, y_val)
                test_data = (x_test, y_test)

                client_instance = LGBMClient(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    num_train=len(x_train),
                    num_val=len(x_val),
                    num_test=len(x_test),
                    model_func=model_func,
                    model_params = model_params
                ).to_client()

                return client_instance
        
        else:
            raise ValueError("Unsupported model type provided to client function.")
    
    # If differential privacy is enabled, wrap `client_fn` in `ClientApp` with LocalDpMod
    if differential_privacy and model_name in ['simple_LSTM', 'stacked_LSTM', 'bidirectional_LSTM', 'GRU']:
        local_dp_obj = LocalDpMod(
            clipping_norm=1.0,
            sensitivity=2.0,
            epsilon=1.5,
            delta=1e-5,
        )
        
        client = ClientApp(client_fn=client_fn, mods=[local_dp_obj])
        return client
    
    else:
        client = ClientApp(client_fn=client_fn)
        return client
