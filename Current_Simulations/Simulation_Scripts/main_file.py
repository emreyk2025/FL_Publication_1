# Centralized and Federated Simulations of the Selected Data sets

import pandas as pd
import dask.dataframe as dd
from models import get_LSTM_Simple, get_LSTM_stacked, get_GRU, get_LSTM_Bidirectional, get_XGBoost, get_LightGBM
from Input_prep import train_val_test
from run import run_simulations
import os


def main():

    # Get the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = ['federated', 'centralized']
    #scenarios = ['centralized', 'federated']
    
    VERBOSE = 1
    # Num. of clients for federated setting:
    num_clients = 100

    # Num. of rounds of federated training:
    num_rounds = 10

    # Differential Privacy
    differential_privacy = True    # True or False

    models = {
        'simple_LSTM': get_LSTM_Simple, 
        'stacked_LSTM': get_LSTM_stacked,
        'bidirectional_LSTM': get_LSTM_Bidirectional,
        'GRU': get_GRU,
        #'XGBoost': get_XGBoost,        
        #'LightGBM': get_LightGBM           
    }

    # There are issues with xgboost and lgbm, will work on it later or will discard it alltogether if too difficult
    # The model parameters serialization/deserialization and fed. aggregation are not straightforward for these models.

    # main_file.py
    model_hyperparams = {
        'simple_LSTM': {'learning_rate': 0.01},
        'stacked_LSTM': {'learning_rate': 0.01},
        'bidirectional_LSTM': {'learning_rate': 0.01},
        'GRU': {'learning_rate': 0.01},
        'XGBoost': {
            'objective': 'reg:squarederror',
            'eta': 0.05,
            'max_depth': 8,
            'eval_metric': ['rmse', 'mae'],
            'num_parallel_tree': 1,
            'subsample': 1,
            'colsample_bytree': 0.8,
        },
        'LightGBM': {
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 8,
            'num_leaves': 30,
            'colsample_bytree': 0.8,
        },
    }


    # Create a list to store all results
    results_list = []

    # Configuration dictionary for datasets
    datasets_info = [
        {
            'name': 'smartmeter',
            'data_path': os.path.join(base_dir, 'Datasets', 'smartmeter_preprocessed_data.csv'),
            'customer_id': 'LCLid',
            'target_column': 'KWH/hh (per hour)',
            'data_loader': 'pd.read_csv',
            'all_meters_func': lambda df: df['LCLid'].nunique(),
            'selected_meters': 100,
            'all_meters': None,
            'folder_path': None,
            'is_dask': False
        },
        {
            'name': 'METER',
            'data_path': os.path.join(base_dir, 'Datasets', 'meter_electricity_preprocessed_1min.csv'),
            'customer_id': 'Meta_idMeta',
            'target_column': 'kW',
            'data_loader': 'pd.read_csv',
            'all_meters_func': lambda df: df['Meta_idMeta'].nunique(),
            'selected_meters': 100,
            'all_meters': None,
            'folder_path': None,
            'is_dask': False
        },
        {
            'name': 'sgsc',
            'data_path': os.path.join(base_dir, 'Datasets', 'sgsc_electricity_use_hourly.csv'),
            'customer_id': 'CUSTOMER_ID',
            'target_column': ' GENERAL_SUPPLY_KWH',
            'data_loader': 'dd.read_csv',
            'all_meters_func': lambda df: df['CUSTOMER_ID'].nunique(),
            'selected_meters': 100,
            'all_meters': None,
            'folder_path': None,
            'is_dask': True
        },
        {
            'name': 'goiener',
            'data_path': None,
            'customer_id': None,
            'target_column': 'kWh',
            'data_loader': None,  # Data is in a folder
            'all_meters': 12149,  # Total number of CSV files
            'selected_meters': 100,
            'folder_path': os.path.join(base_dir, 'Datasets', 'imp_preprocessed', 'pre_pandemic_preprocessed'),
            'is_dask': False
        }
    ]

    for dataset_info in datasets_info:

        dataset_name = dataset_info['name']

        if dataset_info['data_loader'] == 'pd.read_csv':
            data = pd.read_csv(dataset_info['data_path'])
        elif dataset_info['data_loader'] == 'dd.read_csv':
            data = dd.read_csv(dataset_info['data_path'])
        else:
            data = None  # For 'goiener', data is handled differently

        # Compute Dask DataFrame in the beginning before making other operations.
        if dataset_info['name'] == 'sgsc':
            data = data.compute()

        customer_id = dataset_info['customer_id']
        target_column = dataset_info['target_column']
        selected_meters = dataset_info['selected_meters']
        folder_path = dataset_info['folder_path']

        # Determine all_meters
        if dataset_info['name'] == 'goiener':
            all_meters = dataset_info['all_meters']
        else:
            all_meters = dataset_info['all_meters_func'](data)

        # Use all meters or choose a desired number of meters:

        num_meters = all_meters
        #num_meters = selected_meters

        for scenario in scenarios:
            if scenario == 'centralized':
                # Prepare train, validation, test split for centralized scenario
                splitted_dataset = train_val_test(
                    data=data,
                    folder_path=folder_path,
                    scenario='centralized',
                    customer_id=customer_id,
                    meters=selected_meters,
                    all_meters = all_meters
                )

            elif scenario == 'federated':
                # Prepare train, validation, test split for federated scenario
                splitted_dataset = train_val_test(
                    data=data,
                    folder_path=folder_path,
                    scenario='federated',
                    customer_id=customer_id,
                    meters=selected_meters,
                    clients=num_clients,
                    all_meters = all_meters
                )
               
            else:
                continue  


            for model_name, model_func in models.items():
                print(f'##############  CURRENT RUN, DATASET: {dataset_name}, SCENARIO: {scenario}, MODEL: {model_name}  ###############')

                model_params = model_hyperparams[model_name]
                
                if scenario == 'centralized':
                    results_got, history_values, training_time_took, number_of_meters = run_simulations(
                        df_dataset=splitted_dataset, 
                        num_clients=None, 
                        num_meters=num_meters, # choose all_meters or selected_meters here
                        verbose=VERBOSE, 
                        scenario=scenario, 
                        fraction_evaluate=None, 
                        fraction_fit=None, 
                        min_evaluate_clients=None, 
                        model_func=model_func,
                        model_name=model_name,
                        time_steps=30, 
                        target_column=target_column,
                        initial_paramters=None,
                        differential_privacy=differential_privacy,
                        num_rounds=None,
                        model_params = model_params
                    )
                    # Append results to the list
                    results_list.append({
                        'dataset': dataset_name,
                        'scenario': scenario,
                        'model': model_name,
                        'results': results_got,
                        'history': history_values,
                        'training_time': training_time_took,
                        'num_meters': number_of_meters,
                        'num_clients': None
                    })
                elif scenario == 'federated':
                    history_values, training_time_took, number_of_meters, number_of_clients = run_simulations(
                        df_dataset=splitted_dataset, 
                        num_clients=num_clients, 
                        num_meters=num_meters,
                        verbose=VERBOSE, 
                        scenario=scenario, 
                        fraction_evaluate=0.1,
                        initial_paramters=None, 
                        fraction_fit=0.9, 
                        min_evaluate_clients=2, #10
                        model_func=model_func,
                        model_name=model_name,
                        time_steps=30, 
                        target_column=target_column,
                        num_rounds=num_rounds,
                        model_params = model_params,
                        differential_privacy=differential_privacy
                    )
                    results_list.append({
                        'dataset': dataset_name,
                        'scenario': scenario,
                        'model': model_name,
                        'results': None,
                        'history': history_values,
                        'training_time': training_time_took,
                        'num_meters': number_of_meters,
                        'num_clients': number_of_clients
                    })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(base_dir, 'model_results.csv'), index=False)

main()