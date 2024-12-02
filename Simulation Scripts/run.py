import flwr as fl
import time
from tensorflow import convert_to_tensor
import numpy as np
from client_app import get_client_fn
import xgboost as xgb
import tensorflow as tf
from flwr.simulation import run_simulation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_sequences
import lightgbm as lgb
from flwr.server import ServerApp
from client_app import get_client_fn
from server_app import get_server_fn
from flwr.client import ClientApp




def run_simulations(df_dataset, 
                   num_clients, 
                   num_meters, 
                   verbose,
                   num_rounds, 
                   scenario, 
                   target_column, 
                   fraction_fit, 
                   fraction_evaluate, 
                   min_evaluate_clients,
                   initial_paramters, 
                   model_func,
                   differential_privacy, 
                   model_name,
                   model_params, 
                   time_steps = 30):
    
    start_time = time.time()


    if scenario == 'centralized':
        
        # Common data preparation
        data_splits = {
                'train': (df_dataset["train"].drop(columns=[target_column]),
                        df_dataset["train"][target_column]),
                'validation': (df_dataset["validation"].drop(columns=[target_column]),
                            df_dataset["validation"][target_column]),
                'test': (df_dataset["test"].drop(columns=[target_column]),
                        df_dataset["test"][target_column])
            }
        
        # Convert all data to numpy arrays
        for split in data_splits:          
            data_splits[split] = (np.asarray(data_splits[split][0]),
                                np.asarray(data_splits[split][1]))
            
        # Extract features and labels for training, validation, and testing
        x_train, y_train = data_splits['train']
        x_val, y_val = data_splits['validation']
        x_test, y_test = data_splits['test']
        
        
        if model_name in ['simple_LSTM', 'stacked_LSTM', 'bidirectional_LSTM', 'GRU']:

            try:
                # Create sequences
                x_train_seq, y_train_seq = create_sequences(
                    X = x_train, y = y_train, time_steps = time_steps)
                x_val_seq, y_val_seq = create_sequences(
                    X = x_val, y = y_val, time_steps = time_steps)
                x_test_seq, y_test_seq = create_sequences(
                    X = x_test, y = y_test, time_steps = time_steps)

                num_features = x_train_seq.shape[2]

                # Initialize model with time_steps and num_features
                model = model_func(time_steps=time_steps, num_features=num_features, **model_params)

                # Convert data to tensors
                x_train_tensor = convert_to_tensor(x_train_seq, dtype=tf.float32)
                y_train_tensor = convert_to_tensor(y_train_seq, dtype=tf.float32)
                x_val_tensor = convert_to_tensor(x_val_seq, dtype=tf.float32)
                y_val_tensor = convert_to_tensor(y_val_seq, dtype=tf.float32)
                x_test_tensor = convert_to_tensor(x_test_seq, dtype=tf.float32)
                y_test_tensor = convert_to_tensor(y_test_seq, dtype=tf.float32)

                # Add early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                )

                
                # Train the model
                history = model.fit(
                    x_train_tensor,
                    y_train_tensor,
                    validation_data=(x_val_tensor, y_val_tensor),
                    epochs=10,
                    batch_size=512,
                    verbose=verbose,
                    callbacks=[early_stopping]
                )
                

                training_time = time.time() - start_time
                results = model.evaluate(x_test_tensor, y_test_tensor, verbose=verbose)
            

                return results, history.history, training_time, num_meters
        
            except Exception as e:
                print(f"Error in LSTM/GRU (Centralized Scenario) processing: {str(e)}")
                raise

        elif model_name in ['XGBoost', 'LightGBM']:


            if model_name == 'XGBoost':
                
                
                train_DMatrix = xgb.DMatrix(x_train, label = y_train)
                val_DMatrix = xgb.DMatrix(x_val, label = y_val)
                test_DMatrix = xgb.DMatrix(x_test, label = y_test)

                try:

                    model = model_func().train(
                        model_params,
                        train_DMatrix,
                        num_boost_round=300,
                        evals=[(val_DMatrix, "validate"), (train_DMatrix, "train")]
                    )

                    y_pred = model.predict(test_DMatrix)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    print("\nFinal XGBoost Metrics:")
                    print(f"MSE: {mse:.4f}")
                    print(f"MAE: {mae:.4f}")

                    results = [mse, mae]

                    training_time = time.time() - start_time

                    return results, None, training_time, num_meters
                
                except Exception as e:
                    print(f"Error in XGBoost (Centralized Scenario) processing: {str(e)}")
                    raise
                    
            elif model_name == 'LightGBM':

                model = model_func(model_params)
                eval_results = {}
                
                try:

                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_train, y_train), (x_val, y_val)],
                        eval_metric=['mse', 'mae'],callbacks=[
                            lgb.log_evaluation(period=1),  
                            lgb.record_evaluation(eval_results)  
                        ]
                    )

                    y_pred = model.predict(x_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    print("\nFinal LightGBM Metrics:")
                    print(f"MSE: {mse:.4f}")
                    print(f"MAE: {mae:.4f}")
                    results = [mse, mae]

                    training_time = time.time() - start_time

                    return results, None, training_time, num_meters
                
                except Exception as e:
                    print(f"Error in LightGBM (Centralized Scenario) processing: {str(e)}")
                    raise
        
    elif scenario == 'federated':
 
        backend_config = {"client_resources": {"num_cpus": 4.0, "num_gpus": 1.0}}
            
        try:

            server_fn = get_server_fn(fraction_fit=fraction_fit, 
                                    fraction_evaluate=fraction_evaluate, 
                                    min_evaluate_clients=min_evaluate_clients,
                                    initial_parameters=initial_paramters,
                                    model_name=model_name,
                                    num_clients=num_clients,
                                    num_rounds=num_rounds)
            
            server = ServerApp(server_fn=server_fn)
            
            client = get_client_fn(splitted_data=df_dataset, 
                                    model_name=model_name, 
                                    model_func=model_func,
                                    target_column=target_column,
                                    differential_privacy=differential_privacy,
                                    model_params=model_params)
            
            if not differential_privacy:
                client = ClientApp(client_fn=client)

            # Start simulation
            history = run_simulation(server_app=server,
                                     client_app=client,
                                     num_supernodes=num_clients,
                                     backend_config=backend_config)

            end_time = time.time()
            training_time = end_time - start_time
            
            return history, training_time, num_meters, num_clients
        
        except Exception as e:
            print(f"Error in {model_name} Federated Scenario, processing: {str(e)}")
            raise

'''
Old comments

# xgb.DMatrix and lgb.Dataset are special data structures provided by XGBoost and LightGBM, respectively, and they offer several
# speed and memory advantages.

# Why Dont Tree-Based Models Have history.history?
# The history.history attribute is specific to Keras models in deep learning, which store the loss and metrics for each epoch during training.



'''