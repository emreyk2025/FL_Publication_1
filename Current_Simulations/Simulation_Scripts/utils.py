import numpy as np
from typing import Tuple, List
from flwr.common import Metrics

# Define create_sequences function
def create_sequences(X, y, time_steps=1):

    if len(X) <= time_steps:
        raise ValueError(f"Input length ({len(X)}) must be greater than time_steps ({time_steps})")

    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    
    X_seq = np.array(Xs, dtype=np.float32)
    y_seq = np.array(ys, dtype=np.float32)

    # Verify shape is correct
    if len(X_seq.shape) != 3:
        raise ValueError(f"Expected 3D shape (samples, time_steps, features), got shape {X_seq.shape}")


    return X_seq, y_seq


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:                                                         
    # Multiply accuracy of each client by number of examples used                           
    accuracies = [num_examples * m["mae"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(accuracies) / sum(examples)}



def evaluate_metrics_aggregation(eval_metrics):
    # Return an aggregated evaluation metric
    total_num = sum([num for num, _ in eval_metrics])
    mse_aggregated = (
        sum([metrics["MSE"]*num for num, metrics in eval_metrics]) / total_num 
    )
    
    metrics_aggregated = {"MSE": mse_aggregated}
    return metrics_aggregated


'''
Old comments

# Don't be confused here: When you look at the output of client's evaluate method you will see that it has three outputs: results[0] (this is loss)
# number of examples, and accuracy (mae). However, here when we loop through metrics the first element is number of examples as int and
# the metrics dictionary containing accuracy. This is how it is done with Flower: Look at the link: 
# https://medium.com/@adam.narozniak/federated-learning-with-flower-and-tensorflow-d39e2d04f551

# List[Tuple[int, Metrics]] means that the parameter metrics is a
# Tuple containing an integer, number of examples
# a client has used and metrics containing evaluation
# metrics from client accuracy etc.
'''