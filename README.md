# Federated Learning for Smart Cities: Forecasting Short-term Residential Energy Consumption

This project aims to highlight the main advantages of federated learning in short-term energy consumption forecasting in residential buildings. We use the traditional centralized learning setting as the benchmark and assess the effectiveness of federated learning in privacy protection, computational efficiency, and predictive performance. In addition, local differential privacy adds an additional level of privacy protection by clipping model updates and adding noise to the parameters before sending them to the server. Our simulation scenarios include several state-of-the-art machine-learning models and diverse real-world energy consumption datasets from multiple cities around the world.

## Datasets

- Smart Grid Smart City Customer Trial Data - Australia
- GoiEner Dataset - Spain
- METER UK Household Electricity and Activity Survey - United Kingdom
- SmartMeter Energy Consumption Data in London Households - United Kingdom



| **Dataset**       | **Time Period**       | **Time Intervals**       | **Size of the Data**       |
|:-------------------:|:------------------:|:------------------:|:------------------:|
| Smart Grid Smart City Customer Trial Data       | 2010 - 2014      | 30 min.      | 18.8 GB      |
| SmartMeter Energy Consumption Data in London Households       | 2011 - 2014      | 30 min      | 10 GB      |
| GoiEner Data Set       | 2014 - 2022      |  Hourly     | 20.5 GB      |
| METER UK Household Electricity and Activity Survey       | 2016 - 2019      |  Minutely      | 22 MB      |



## Models

- Simple LSTM
- Stacked LSTM
- Bidirectional LSTM
- GRU
- XGBoost
- LightGBM

## Code Structure

This GitHub repository provides the simulation pipeline to run each forecasting model on both centralized and federated learning settings using preprocessed datasets iteratively. The Previous_Simulations_SmartMeter_Dataset contains the simulation and data preprocessing code for SmartMeter London Dataset. On the other hand, Current_Simulations folder includes the python files and jupyter notebooks for running the simulations for 4 different electricity consumption datasets mentioned above. The Python code is prepared in a modular structure where individual files are imported and used in the main file. The data preprocessing notebooks are stored in the Datasets folder. The preprocessed datasets can be accessed using the Google Drive link below:

https://drive.google.com/drive/folders/1vvCZaxM0p65zaKHAp7656IoOSP4xvQ-5?usp=sharing

A summary table for the identified datasets:

https://docs.google.com/spreadsheets/d/1ITpv8mRyQ7N0JAh97GfTNjc3XqFATZcrDcUY_fXbyMM/edit?usp=sharing

The rest of the files can be found in the "Simulation Scripts" folder.

