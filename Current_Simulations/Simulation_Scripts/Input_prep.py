# Some notes, since we are working with time-series data the test set should be the most recent part of data.
# This is reflected in the code as we split the data in a way that first 80 % of observations are for training, 
# between 80 % and 90 % are for validation and the rest for test.

import pandas as pd
import os
import random
import dask.dataframe as dd
import math

def train_val_test(data=None, folder_path=None, scenario=None, customer_id=None, meters=None, all_meters = None, clients=None):
    random.seed(25)

    if meters is not None and (not isinstance(meters, int) or meters <= 0):
        raise ValueError("Number of meters must be a positive integer.")

    # Determine data source
    if data is not None and customer_id is not None:
        # Handle Dask DataFrames if necessary
        if isinstance(data, dd.DataFrame):
            # unique customers is a list of customer ids.
            unique_customers = data[customer_id].unique().compute()
        else:
            unique_customers = data[customer_id].unique()
        total_meters_available = len(unique_customers)
        if meters is None or meters > total_meters_available:
            meters = total_meters_available
        selected_customers = random.sample(list(unique_customers), meters)
        # Filter data to include only selected customers
        if isinstance(data, dd.DataFrame):
            data = data[data[customer_id].isin(selected_customers)].compute()
        else:
            data = data[data[customer_id].isin(selected_customers)]
    elif folder_path is not None:
        customer_ids = []
        total_meters_available = meters
        for file_name in os.listdir(folder_path):
            customer_id = file_name.split(".")[0]
            customer_ids.append(customer_id)
        selected_customer_ids = random.sample(customer_ids, meters)
    else:
        raise ValueError("Either data and customer_id or folder_path must be provided.")
    

    if scenario == 'centralized':
        
        # Initialize lists to collect DataFrames
        train_list = []
        val_list = []
        test_list = []

        train = pd.DataFrame()
        val = pd.DataFrame()
        test = pd.DataFrame()

        if data is not None:
            # Data is in a DataFrame
            for customer in selected_customers:
                tmp_data = data[data[customer_id] == customer]
                val_split = int(len(tmp_data) * 0.8)
                test_split = int(len(tmp_data) * 0.9)

                train_ = tmp_data.iloc[:val_split]
                val_ = tmp_data.iloc[val_split:test_split]
                test_ = tmp_data.iloc[test_split:]

                # Inside the loop
                train_list.append(train_)
                val_list.append(val_)
                test_list.append(test_)

        elif folder_path is not None:
            # Data is in files
            for file_name in selected_customer_ids:
                current_customers_directory = os.path.join(folder_path, file_name)
                tmp_data = pd.read_csv(current_customers_directory + ".csv")
                val_split = int(len(tmp_data) * 0.8)
                test_split = int(len(tmp_data) * 0.9)

                train_ = tmp_data.iloc[:val_split]
                val_ = tmp_data.iloc[val_split:test_split]
                test_ = tmp_data.iloc[test_split:]

                # Inside the loop
                train_list.append(train_)
                val_list.append(val_)
                test_list.append(test_)
        
        else:
            raise ValueError("No data provided for centralized scenario.")
        
        train = pd.concat(train_list, ignore_index=True)
        val = pd.concat(val_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)

        return {"train": train, "validation": val, "test": test}

    elif scenario == 'federated':
        
        if clients is None or not isinstance(clients, int) or clients <= 0:
            raise ValueError("Number of clients must be a positive integer.")

        train_l = []
        val_l = []
        test_l = []

        part_counter = 0
        meters_per_client = int(math.ceil(meters / clients))
        part_id = 0

        if data is not None:
            # Data is in a DataFrame

            for customer in selected_customers:

                # Get the data for the current meter:
                tmp_data = data[data[customer_id] == customer]

                # Split index between train and validate
                val_split = int(len(tmp_data) * 0.8)
                # Split index between validate and test
                test_split = int(len(tmp_data) * 0.9)

                # Set initial splits for current meter
                train_ = tmp_data[:val_split]
                valid_ = tmp_data[val_split:test_split]
                test_ = tmp_data[test_split:]
                
                

                # Concatanate train and validation inside current partition  
                if (part_counter > 0):
                    train = pd.concat([train, train_], ignore_index=True)
                    valid = pd.concat([valid, valid_], ignore_index=True)
                    test = pd.concat([test, test_], ignore_index=True)
                    

                else: 
                    train = train_
                    valid = valid_
                    test = test_


                # Increase the counter inside current partition 
                part_counter += 1
                
                if (part_counter >= meters_per_client) and (part_id < clients - 1):

                    # Reset counter inside current partition for the next one 
                    part_counter = 0

                    # Append validation and train data to the partition they belong to   
                    train_l.append(train.reset_index(drop=True))
                    val_l.append(valid.reset_index(drop=True))
                    test_l.append(test.reset_index(drop=True))

                    

                    # Increase partition counter
                    part_id += 1

                    # Reset values 
                    train = pd.DataFrame()
                    valid = pd.DataFrame()
                    test = pd.DataFrame()
            
            if (part_counter != 0):
                # Append train and validation to the last partition
                train_l.append(train)
                val_l.append(valid)
                test_l.append(test)                

                return {"train": train_l, "validation": val_l, "test": test_l}


            
        elif folder_path is not None:

            part_counter = 0
            meters_per_client = int(math.ceil(meters / clients))
            part_id = 0
            
            for file_name in selected_customer_ids:
                
                current_customers_directory = os.path.join(folder_path, file_name + ".csv")

                # Get the data for the current meter:
                tmp_data = pd.read_csv(current_customers_directory)

                # Split index between train and validate
                val_split = int(len(tmp_data) * 0.8)
                # Split index between validate and test
                test_split = int(len(tmp_data) * 0.9)

                # Set initial splits for current meter
                train_ = tmp_data[:val_split]
                valid_ = tmp_data[val_split:test_split]
                test_ = tmp_data[test_split:]

                # Concatanate train and validation inside current partition  
                if (part_counter > 0):
                    train = pd.concat([train, train_], ignore_index=True)
                    valid = pd.concat([valid, valid_], ignore_index=True)
                    test = pd.concat([test, test_], ignore_index=True)

                else: 
                    train = train_
                    valid = valid_
                    test = test_

                # Increase the counter inside current partition 
                part_counter += 1
                if (part_counter >= meters_per_client) and (part_id < clients - 1):

                    # Reset counter inside current partition for the next one 
                    part_counter = 0

                    # Append validation and train data to the partition they belong to   
                    train_l.append(train.reset_index(drop=True))
                    val_l.append(valid.reset_index(drop=True))
                    test_l.append(test.reset_index(drop=True))

                    # Increase partition counter
                    part_id += 1

                    # Reset values 
                    train = pd.DataFrame()
                    valid = pd.DataFrame()
                    test = pd.DataFrame()
            
            if (part_counter != 0):
                # Append train and validation to the last partition
                train_l.append(train.reset_index(drop=True))
                val_l.append(valid.reset_index(drop=True))
                test_l.append(test.reset_index(drop=True))

                return {"train": train_l, "validation": val_l, "test": test_l}

        else:
            raise ValueError("No data provided for federated scenario.")

    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    




    '''
    old for loop for federated scenario:
    for client_id in range(clients):
                client_train = pd.DataFrame()
                client_val = pd.DataFrame()
                client_test = pd.DataFrame()
                num_meters_in_client = meters_per_client + (1 if client_id < extra_meters else 0)
                for _ in range(num_meters_in_client):
                    if customer_index >= len(selected_customers):
                        break
                    customer = selected_customers[customer_index]
                    customer_index += 1
                    tmp_data = data[data[customer_id] == customer]
                    val_split = int(len(tmp_data) * 0.8)
                    test_split = int(len(tmp_data) * 0.9)

                    train_ = tmp_data.iloc[:val_split]
                    val_ = tmp_data.iloc[val_split:test_split]
                    test_ = tmp_data.iloc[test_split:]

                    client_train = pd.concat([client_train, train_], ignore_index=True)
                    client_val = pd.concat([client_val, val_], ignore_index=True)
                    client_test = pd.concat([client_test, test_], ignore_index=True)



                    # Data is in files
            file_index = 0
            for client_id in range(clients):
                client_train = pd.DataFrame()
                client_val = pd.DataFrame()
                client_test = pd.DataFrame()
                num_meters_in_client = meters_per_client + (1 if client_id < extra_meters else 0)
                for _ in range(num_meters_in_client):
                    if file_index >= len(selected_files):
                        break
                    file_name = selected_files[file_index]
                    file_index += 1
                    current_customers_directory = os.path.join(folder_path, file_name)
                    tmp_data = pd.read_csv(current_customers_directory)
                    val_split = int(len(tmp_data) * 0.8)
                    test_split = int(len(tmp_data) * 0.9)

                    train_ = tmp_data.iloc[:val_split]
                    val_ = tmp_data.iloc[val_split:test_split]
                    test_ = tmp_data.iloc[test_split:]

                    client_train = pd.concat([client_train, train_], ignore_index=True)
                    client_val = pd.concat([client_val, val_], ignore_index=True)
                    client_test = pd.concat([client_test, test_], ignore_index=True)
                train_l.append(client_train.reset_index(drop=True))
                val_l.append(client_val.reset_index(drop=True))
                test_l.append(client_test.reset_index(drop=True))
    '''





'''
for customer in selected_customers:

                # Get the data for the current meter:
                tmp_data = data[data[customer_id] == customer]

                # Split index between train and validate
                val_split = int(len(tmp_data) * 0.8)
                # Split index between validate and test
                test_split = int(len(tmp_data) * 0.9)

                # Set initial splits for current meter
                train_ = tmp_data[:val_split]
                valid_ = tmp_data[val_split:test_split]
                test_ = tmp_data[test_split:]

                # Concatanate train and validation inside current partition  
                if (part_counter > 0):
                    train = pd.concat([train, train_], ignore_index=True)
                    valid = pd.concat([valid, valid_], ignore_index=True)
                    test = pd.concat([test, test_], ignore_index=True)

                else: 
                    train = train_
                    valid = valid_
                    test = test_

                # Increase the counter inside current partition 
                part_counter += 1
                if (part_counter >= meters_per_client) and (part_id < clients - 1):

                    # Reset counter inside current partition for the next one 
                    part_counter = 0

                    # Append validation and train data to the partition they belong to   
                    train_l.append(train.reset_index(drop=True))
                    val_l.append(valid.reset_index(drop=True))
                    test_l.append(test.reset_index(drop=True))

                    # Increase partition counter
                    part_id += 1

                    # Reset values 
                    train = pd.DataFrame()
                    valid = pd.DataFrame()
                    test = pd.DataFrame()
            
            if (part_counter != 0):
                # Append train and validation to the last partition
                train_l.append(train.reset_index(drop=True))
                val_l.append(valid.reset_index(drop=True))
                test_l.append(test.reset_index(drop=True))

                return {"train": train_l, "validation": val_l, "test": test_l}

'''