import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import convert_to_tensor

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

# Define the stacked LSTM model 
def get_model_stacked():
    model = Sequential()
    model.add(Input(shape=(None, 1)))
    model.add(LSTM(20, return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(LSTM(15, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(5))  
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Define the simple LSTM model
def get_model_simple():
    model = Sequential()
    model.add(Input(shape=(None, 1)))
    model.add(LSTM(50)) 
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, mdl) -> None:
        # Create model
        self.model = mdl
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        #batch_size: int = config["batch_size"]
        #epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size = 512,
            epochs = 10,
            verbose=0
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["mean_absolute_error"][0]
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        #steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        results = self.model.evaluate(self.x_val, self.y_val, 32)#, steps=steps)
        num_examples_test = len(self.x_val)
        return results[0], num_examples_test, {"accuracy": results[1]}

# Function that prepares the input for Centralised models
def train_test_validate_centr(data, meters):

    for i in range(0, meters):

        # Get the data for current meter
        tmp_data = data[data['LCLid'] == i] 
        
        val_split = int(len(tmp_data) * 0.8)
        test_split = int(len(tmp_data) * 0.9)

        # Set initial splits for current meter
        train_ = tmp_data[:val_split]
        vali_ = tmp_data[val_split:test_split]
        test_ = tmp_data[test_split:]

        # Concatanate the test data 
        if (i > 0):
            train = pd.concat([train, train_], ignore_index=True)
            valid = pd.concat([valid, vali_], ignore_index=True)
            test = pd.concat([test, test_], ignore_index=True)
        else:
            train = train_
            valid = vali_
            test = test_

    return {"train": pd.DataFrame(train, columns=data.columns), 
            "test": pd.DataFrame(test, columns=data.columns), 
            "validation": pd.DataFrame(valid, columns=data.columns)} 

# Function that prepares the input for Federated models
def train_test_validate(data, meters, clients):

    train_list = [None] * clients
    val_list = [None] * clients
    test = pd.DataFrame()

    client_counter = 0
    partition_counter = 0

    for i in range(0, meters):
            # Get the data for current meter
            tmp_data = data[data['LCLid'] == i]

            # Split index between train and validate
            val_split = int(len(tmp_data) * 0.8)
            # Split index between validate and test
            test_split = int(len(tmp_data) * 0.9)

            # Set initial splits for current meter
            train_ = tmp_data[:val_split]
            vali_ = tmp_data[val_split:test_split]
            test_ = tmp_data[test_split:]

            # Concatanate the test data 
            if (i > 0):
                test = pd.concat([test, test_], ignore_index=True)
            else:
                test = test_

            # Concatanate train and validation inside current partition  
            if (partition_counter == 0):
              train_list[client_counter] = [train_]
              val_list[client_counter] = [vali_]
            else:
              train_list[client_counter][0] = pd.concat([train_list[client_counter][0], train_], ignore_index=True)
              val_list[client_counter][0] = pd.concat([val_list[client_counter][0], vali_], ignore_index=True)
            
            client_counter += 1
            if (client_counter >= clients):
                client_counter = 0
                partition_counter += 1
        
    return {"train": train_list, "test": test.reset_index(drop=True), "validation": val_list}


def get_client_fn(partition, model):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        
        # Extract partition for client with id = cid
        trainset = partition["train"][int(cid)][0]
        valset = partition["validation"][int(cid)][0] 

        # Split into features and targets and transform into tensors 
        x_train_tensor = convert_to_tensor(np.asarray(trainset.drop(columns=['KWH/hh (per hour) '])).astype(np.float32))
        y_train_tensor = convert_to_tensor(np.asarray(trainset['KWH/hh (per hour) ']).astype(np.float32))
        x_val_tensor = convert_to_tensor(np.asarray(valset.drop(columns=['KWH/hh (per hour) '])).astype(np.float32))
        y_val_tensor = convert_to_tensor(np.asarray(valset['KWH/hh (per hour) ']).astype(np.float32))

        # Create and return client
        return FlowerClient(x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, model).to_client()

    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset, verbose, model):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        
        model.set_weights(parameters)  # Update model with the latest parameters
        results = model.evaluate(convert_to_tensor(np.asarray(testset.drop(columns=['KWH/hh (per hour) '])).astype(np.float32)), 
                                 convert_to_tensor(np.asarray(testset['KWH/hh (per hour) ']).astype(np.float32)), 
                                 verbose=verbose)
        return results[0], {"accuracy": results[1]}

    return evaluate

# Plots results of the Federated models
def plots_of_simulation_fed(history, scenario, num_meters, num_clients=0, training_time=None):
    log.write(f"{scenario} --> metrics_distributed {np.average(history.metrics_distributed['accuracy'])} // losses_distributed {np.average(history.losses_distributed)} // metrics_centralized {np.average(history.metrics_centralized['accuracy'])} // losses_centralized {np.average(history.losses_centralized)} \n")
    global_accuracy_centralised = history.metrics_distributed["accuracy"]
    global_loss_centralised = history.losses_distributed
    rounds = [data[0] for data in global_accuracy_centralised]
    loss = [data[1] for data in global_loss_centralised]
    acc = [data[1] for data in global_accuracy_centralised]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("MEAN ABSOLUTE ERROR", "MEAN SQUARED ERROR"))

    # Add scatter plot for accuracy
    fig.add_trace(go.Scatter(x=rounds, y=acc, mode='markers', name='MAE'), row=1, col=1)
    # Add line plot for accuracy
    fig.add_trace(go.Scatter(x=rounds, y=acc, mode='lines', name='MAE Line'), row=1, col=1)

    # Add scatter plot for loss
    fig.add_trace(go.Scatter(x=rounds, y=loss, mode='markers', name='MSE'), row=2, col=1)
    # Add line plot for loss
    fig.add_trace(go.Scatter(x=rounds, y=loss, mode='lines', name='MSE Line'), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800,  # Height of the figure
        title_text=f"FEDERATED SMART METERS - {num_meters} clients with {num_clients} sampled clients per round\nTraining time: {training_time:.2f} seconds",
    )

    # Update x-axis for all subplots
    fig.update_xaxes(title_text="Round", row=2, col=1)
    # Update y-axis for each subplot
    fig.update_yaxes(title_text="MEAN ABSOLUTE ERROR", row=1, col=1)
    fig.update_yaxes(title_text="MEAN SQUARED ERROR", row=2, col=1)

    # Save the plot into HTML file 
    name = f'{scenario}_METERS{num_meters}_CLIENTS{num_clients}'
    fig.write_html(f'images/{name}.html')

# Plots results of the centralised models
def plots_of_simulation_cen(history, scenario, num_meters, training_time=None):
    rounds = np.arange(0,len(history.history['loss']))
    loss = history.history['loss']
    acc = history.history['mean_absolute_error']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("MEAN ABSOLUTE ERROR", "MEAN SQUARED ERROR"))

    # Add scatter plot for accuracy
    fig.add_trace(go.Scatter(x=rounds, y=acc, mode='markers', name='MAE'), row=1, col=1)
    # Add line plot for accuracy
    fig.add_trace(go.Scatter(x=rounds, y=acc, mode='lines', name='MAE Line'), row=1, col=1)

    # Add scatter plot for loss
    fig.add_trace(go.Scatter(x=rounds, y=loss, mode='markers', name='MSE'), row=2, col=1)
    # Add line plot for loss
    fig.add_trace(go.Scatter(x=rounds, y=loss, mode='lines', name='MSE Line'), row=2, col=1)

    name = f'{scenario}_METERS{num_meters}'.capitalize()

    # Update layout
    fig.update_layout(
        height=800,  # Height of the figure
        title_text=f"{name}\nTraining time: {training_time:.2f} seconds",
    )

    # Update x-axis for all subplots
    fig.update_xaxes(title_text="Round", row=2, col=1)
    # Update y-axis for each subplot
    fig.update_yaxes(title_text="MEAN ABSOLUTE ERROR", row=1, col=1)
    fig.update_yaxes(title_text="MEAN SQUARED ERROR", row=2, col=1)

    # Save into HTML file
    fig.write_html(f'images/{name}.html')


def run_federated(df_dataset, num_clients, num_meters, verbose, scenario, fraction_fit, fraction_evaluate, model, num_rounds=10):
    start_time = time.time()

    

    # Get the whole test set for centralised evaluation
    centralized_testset = df_dataset["test"]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,  # Sample 10% of available clients for training
        fraction_evaluate=fraction_evaluate,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            num_clients * 0.75
        ),  # Wait until at least 75 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset, verbose, model),  # global evaluation function
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(df_dataset, model),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        actor_kwargs={
                "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
                # does nothing if `num_gpus` in client_resources is 0.0
            },
    )
    end_time = time.time()
    training_time = end_time - start_time

    plots_of_simulation_fed(history, scenario, num_meters, num_clients, training_time=training_time)

def run_centralized(df_dataset, num_meters, VERBOSE, scenario, model):
    start_time = time.time()

    if '80' in scenario:
        #randomly drop 20% of columns
        rows_to_drop_training = df_dataset["train"].sample(frac=0.2).index
        df_dataset["train"].drop(rows_to_drop_training, inplace=True)
        df_dataset["train"].reset_index(drop=True, inplace=True)
        rows_to_drop_test = df_dataset["test"].sample(frac=0.2).index
        df_dataset["test"].drop(rows_to_drop_test, inplace=True)
        df_dataset["test"].reset_index(drop=True, inplace=True)
        rows_to_drop_validation = df_dataset["validation"].sample(frac=0.2).index
        df_dataset["validation"].drop(rows_to_drop_validation, inplace=True)
        df_dataset["validation"].reset_index(drop=True, inplace=True)
    elif 'column' in scenario:
        if 'is_weekend' in df_dataset["train"].columns:
            #drop is_weekend column
            df_dataset["train"].drop(columns=['is_weekend'], inplace=True)
            df_dataset["test"].drop(columns=['is_weekend'], inplace=True)
            df_dataset["validation"].drop(columns=['is_weekend'], inplace=True)
    elif 'missing_hour' in scenario:
        # Drop consistently the same hour from all the partitions
        df_dataset["train"] = df_dataset["train"][df_dataset["train"]["hour"] != 18]
        df_dataset["train"].reset_index(drop=True, inplace=True)
        df_dataset["test"] = df_dataset["test"][df_dataset["test"]["hour"] != 18]
        df_dataset["test"].reset_index(drop=True, inplace=True)
        df_dataset["validation"] = df_dataset["validation"][df_dataset["validation"]["hour"] != 18]
        df_dataset["validation"].reset_index(drop=True, inplace=True)

    # Convert data to tensors
    x_train_tensor = convert_to_tensor(np.asarray(df_dataset["train"].drop(columns=['KWH/hh (per hour) '])).astype(np.float32))
    y_train_tensor = convert_to_tensor(np.asarray(df_dataset["train"]['KWH/hh (per hour) ']).astype(np.float32))
    x_val_tensor = convert_to_tensor(np.asarray(df_dataset["validation"].drop(columns=['KWH/hh (per hour) '])).astype(np.float32))
    y_val_tensor = convert_to_tensor(np.asarray(df_dataset["validation"]['KWH/hh (per hour) ']).astype(np.float32))
    x_test_tensor = convert_to_tensor(np.asarray(df_dataset["test"].drop(columns=['KWH/hh (per hour) '])).astype(np.float32))
    y_test_tensor = convert_to_tensor(np.asarray(df_dataset["test"]['KWH/hh (per hour) ']).astype(np.float32))

    history = model.fit(x_train_tensor, y_train_tensor, validation_data=(x_val_tensor, y_val_tensor), epochs=10, batch_size=512, verbose=VERBOSE)
    end_time = time.time()
    training_time = end_time - start_time

    results = model.evaluate(x_test_tensor, y_test_tensor, verbose=VERBOSE)
    log.write(f"{scenario} --> {results} \n")
    
    plots_of_simulation_cen(history, scenario, num_meters, training_time=training_time)

log = open("logging_file.txt","w")

def main():

    enable_tf_gpu_growth()
    scenarios = ['federated', 'centralized_removed_column', 'centralized_missing_hour', 'centralized_80pr_training_data', 'centralized_all_data']

    print("opening dataset")
    dataset = pd.read_csv("Preprocessed_data.csv")

    all_meters = dataset['LCLid'].max() + 1

    scenarios_meters_clients = {
        50: {'clients': [30, 50], 'params': [0.5, 0.1]},
        100: {'clients': [30, 100], 'params': [0.2, 0.1]},
        all_meters: {'clients': [100], 'params': [0.2, 0.1]}
    }

    VERBOSE = 0

    models = {'simple': get_model_simple(), 
              'stacked': get_model_stacked()}
    
    print("Reading centralized")
    df_dataset_centr = train_test_validate_centr(dataset, all_meters)

    print("STARTING")
    for scenario in scenarios:
                for meters, othr in scenarios_meters_clients.items():
                    print(meters)
                    if 'federated' in scenario:
                        for client in othr['clients']:
                          df_dataset = train_test_validate(dataset, client, meters)
                          for model_name, model in models.items():
                              name_scenario = f'{scenario}_{model_name}_{client}_run0'
                              print(name_scenario)
                              run_federated(df_dataset, client, meters, VERBOSE, name_scenario, fraction_fit = othr['params'][0], fraction_evaluate= othr['params'][1], model=model)
                    elif 'centralized' in scenario:
                      df_dataset_c = df_dataset_centr
                      df_dataset_c["train"] = df_dataset_centr["train"][df_dataset_centr["train"]['LCLid'] < meters]
                      df_dataset_c["test"] = df_dataset_centr["test"][df_dataset_centr["test"]['LCLid'] < meters]
                      df_dataset_c["validation"] = df_dataset_centr["validation"][df_dataset_centr["validation"]['LCLid'] < meters]
                      for model_name, model in models.items():
                          name_scenario = f'{scenario}_{model_name}_run0'
                          print(name_scenario)
                          run_centralized(df_dataset_c, meters, VERBOSE, name_scenario, model=model) 

main()

log.close()