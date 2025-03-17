# Import the dependencies
import pandas as pd
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import losses
from keras import layers, Sequential, optimizers
import random

# Set seeds for reproducibility
random.seed(25)
np.random.seed(25)
tf.random.set_seed(25)

# Parameters and Data loading
time_steps = 168
target_column = 'KWH/hh (per hour)'
unique_customer_id = 'LCLid'

# Load the dataset (1.54GB file)
dataset = pd.read_csv("SmartMeter_2013_hourly.csv")

unique_customers = dataset[unique_customer_id].unique()
meter_counts = [50, 100, len(unique_customers)]

# Compute the number of features by dropping the target column.
sample_features = dataset.drop(columns=[target_column, unique_customer_id]).iloc[0].values
num_features = sample_features.shape[0]

# Generator to Create Sequences

def sequence_generator(split, selected_customers, dataset, time_steps, target_column, data_dropout):
    """
    Generator that loops over each customer, splits his/her data, and yields
    sliding-window sequences (X_seq, y_seq) for the requested split.
    
    Parameters:
      - split: one of 'train', 'validation', or 'test'
      - selected_customers: list of customer ids to process
      - dataset: the full DataFrame
      - time_steps: number of time steps per sequence
      - target_column: name of the target column in the DataFrame
      - data_dropout: fraction of sequences to drop (0.0 means use all)
    """
    for customer in selected_customers:
        tmp_data = dataset[dataset[unique_customer_id] == customer]
        # Skip if not enough data for a sequence
        if len(tmp_data) < time_steps + 1:
            continue
        # Reset the index to ensure proper sequential ordering.
        tmp_data = tmp_data.reset_index(drop=True)
        n = len(tmp_data)
        # Define per-customer splits:
        val_split = int(n * 0.8)
        test_split = int(n * 0.9)
        if split == 'train':
            split_data = tmp_data.iloc[:val_split]
        elif split == 'validation':
            split_data = tmp_data.iloc[val_split:test_split]
        elif split == 'test':
            split_data = tmp_data.iloc[test_split:]
        else:
            raise ValueError("Invalid split value: " + split)
        # If not enough data in the split, skip
        if len(split_data) < time_steps + 1:
            continue
        # Convert to NumPy arrays and cast to float32

        '''
        Do not worry about boolean columns, .values.astype(np.float32) will automatically convert them to 0s and 1s.
        '''
        features = split_data.drop(columns=[target_column, unique_customer_id]).values.astype(np.float32)
        targets = split_data[target_column].values.astype(np.float32)
        # Create sliding window sequences
        for i in range(len(features) - time_steps):
            # Randomly drop some sequences based on data_dropout rate
            if data_dropout > 0 and random.random() < data_dropout:
                continue
            X_seq = features[i: i + time_steps]
            y_seq = targets[i + time_steps]
            yield X_seq, y_seq

            '''
            Where concetanation happens:
            Each yield statement in the generator returns one sequence pair at a time. There's no explicit "concatenation" 
            within the function itself; instead, it continuously yields sequences from one customer after another.
            '''

# Function to Create tf.data.Dataset

def create_dataset(split, selected_customers, dataset, time_steps, target_column, data_dropout, batch_size):
    """
    Returns a tf.data.Dataset that yields (X_seq, y_seq) batches for the given split.
    """
    output_signature = (
        tf.TensorSpec(shape=(time_steps, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    '''
    f.data.Dataset.from_generator:
    This function collects every sequence yielded by the generator and automatically concatenates them into one centralized 
    tf.data.Dataset for that split.

    So, the final training dataset contains all training sequences from customer A followed by all training sequences from 
    customer B.The same concept applies when creating the validation or test datasets.
    '''

    ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(split, selected_customers, dataset, time_steps, target_column, data_dropout),
        output_signature=output_signature
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Define the models

def get_LSTM_Simple(time_steps = 168, num_features = 8, learning_rate = 0.01, neurons=60, loss_function='mean_squared_error'):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.LSTM(neurons, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer=optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mean_absolute_error'])
    return model


def get_LSTM_stacked(time_steps = 168, num_features = 8, learning_rate = 0.01, neurons=60, loss_function='mean_squared_error'):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.LSTM(int(neurons*0.8), return_sequences=True)) 
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(int(neurons*0.6), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(int(neurons*0.4), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(int(neurons*0.2), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(int(neurons*0.1), return_sequences=True))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mean_absolute_error'])
    return model


# Create lists for neuron multipliers and loss functions
neurons_mult_list = [2, 4, 10]

loss_functions = {
    'MSE': losses.MeanSquaredError(),
    'Huber (δ=1.5)': losses.Huber(delta=1.5),
    'MSLE': losses.MeanSquaredLogarithmicError()
}
# 'Cross Entropy': losses.BinaryCrossentropy(), # This loss function is not suitable for regression tasks

data_points_drop = [0 ,0.1, 0.3]
models_to_run = ['LSTM_Simple', 'LSTM_stacked']

batch_size = 512
epochs = 5 # Results show little improvement after 5th epoch.

all_metrics = []

# Run the each model in a loop  with the different loss functions and neuron multipliers
for selected_num_meters in meter_counts:
    # For each meter count, randomly select that many customers
    selected_customers = random.sample(list(unique_customers), selected_num_meters)
    
    for neuron_multiplier in neurons_mult_list:
        for loss_function_key, loss_function_value in loss_functions.items():
            for data_dropout in data_points_drop:

                # Precompute the sample counts

                train_gen = sequence_generator('train', selected_customers, dataset, time_steps, target_column, data_dropout)
                train_sample_count = sum(1 for _ in train_gen)
                steps_per_epoch = train_sample_count // batch_size
                if steps_per_epoch == 0:
                    steps_per_epoch = 1  # Ensure at least one step


                # Validation samples
                val_gen = sequence_generator('validation', selected_customers, dataset, time_steps, target_column, 0)
                val_sample_count = sum(1 for _ in val_gen)
                validation_steps = val_sample_count // batch_size
                if validation_steps == 0:
                    validation_steps = 1

                test_gen = sequence_generator('test', selected_customers, dataset, time_steps, target_column, 0)
                test_sample_count = sum(1 for _ in test_gen)
                test_steps = test_sample_count // batch_size
                if test_steps == 0:
                    test_steps = 1  # Ensure at least 1 step


                # Create datasets with .repeat()
                train_ds = create_dataset('train', selected_customers, dataset, time_steps, target_column, data_dropout, batch_size).repeat()
                val_ds = create_dataset('validation', selected_customers, dataset, time_steps, target_column, 0, batch_size).repeat()
                test_ds = create_dataset('test', selected_customers, dataset, time_steps, target_column, 0, batch_size)
                
                for model_name in models_to_run:
                    print(f"\nTraining {model_name} model:")
                    print(f"- Neurons: {60 * neuron_multiplier}")
                    print(f"- Loss function: {loss_function_key}")
                    print(f"- Data dropout: {data_dropout * 100}%")
                    print(f"- Number of meters: {selected_num_meters}")
                    
                    # Instantiate the chosen model
                    if model_name == 'LSTM_Simple':
                        model = get_LSTM_Simple(time_steps=time_steps,
                                                num_features=num_features,
                                                learning_rate=0.01,
                                                neurons=60 * neuron_multiplier,
                                                loss_function=loss_function_value)
                    elif model_name == 'LSTM_stacked':
                        model = get_LSTM_stacked(time_steps=time_steps,
                                                 num_features=num_features,
                                                 learning_rate=0.01,
                                                 neurons=60 * neuron_multiplier,
                                                 loss_function=loss_function_value)
                    else:
                        raise ValueError("Unknown model name.")
                    
                    start_time = time.time()
                    history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=1
                    )
                    end_time = time.time()
                    training_time = end_time - start_time
                    
                    results = model.evaluate(test_ds, steps=test_steps, verbose=1)
                    metrics_data = {
                        'model_name': model_name,
                        'loss': results[0],
                        'mae': results[1],
                        'num_meters': selected_num_meters,
                        'neuron_multiplier': neuron_multiplier,
                        'loss_function': loss_function_key,
                        'data_dropout': data_dropout,
                        'training_time': training_time
                    }
                    
                    print(f"\nTraining completed in {training_time:.2f} seconds")
                    print(f"Test Loss: {results[0]:.4f}")
                    print(f"Test MAE: {results[1]:.4f}")
                    
                    all_metrics.append(metrics_data)
                    # Save intermediate results to CSV
                    pd.DataFrame(all_metrics).to_csv('simulation_metrics.csv', index=False)