# Federated Learning for Smart Cities: Forecasting Short-term Residential Energy Consumption

This project aims to highlight the main advantages of federated learning in short-term energy consumption forecasting in residential buildings. We use the traditional centralized learning setting as the benchmark and assess the effectiveness of federated learning in privacy protection, computational efficiency, and predictive performance. In addition, local differential privacy adds an additional level of privacy protection by clipping model updates and adding noise to the parameters before sending them to the server. Our simulation scenarios include several state-of-the-art machine-learning models and diverse real-world energy consumption datasets from multiple cities around the world.

## Datasets

- Smart Grid Smart City Customer Trial Data - Australia
- GoiEner Dataset - Spain
- METER UK Household Electricity and Activity Survey - UK
- SmartMeter Energy Consumption Data in London Households - UK



| **Dataset**       | **Time Period**       | **Time Intervals**       | **Size of the Data**       |
|:-------------------:|:------------------:|:------------------:|:------------------:|
| Smart Grid Smart City Customer Trial Data       | 2010 - 2014      | 30 min.      | 18.8 GB      |
| GoiEner Data Set       | 2014 - 2022      |  Hourly     | 20.5 GB      |
| METER UK Household Electricity and Activity Survey       | 2016 - 2019      |  Minutely      | 22 MB      |
| SmartMeter Energy Consumption Data in London Households       | 2011 - 2014      | 30 min      | 10 GB      |


## Models

- Simple LSTM
- Stacked LSTM
- Bidirectional LSTM
- GRU
- XGBoost
- LightGBM

## Code Structure

This GitHub repository provides the simulation pipeline to run each forecasting model on both centralized and federated learning frameworks.

project-root/
├── data/
│   ├── raw/                # Placeholder for original datasets
│   ├── processed/          # Placeholder for preprocessed datasets
│   └── loaders/            # Placeholder for data loading and augmentation scripts
├── models/
│   ├── centralized/        # Placeholder for centralized models
│   ├── federated/          # Placeholder for federated models
│   └── utilities/          # Includes shared utilities like `utils.py`
│       └── utils.py        # Uploaded utility file
├── simulations/
│   ├── centralized/        # Placeholder for centralized simulation scripts
│   ├── federated/          # Placeholder for federated simulation scripts
│   ├── comparison.py       # Placeholder for benchmarking scripts
│   ├── main_file.py        # Uploaded main execution script
│   ├── run.py              # Script to run the simulations
│   ├── server_app.py       # Server-side application logic
│   └── client_app.py       # Client-side application logic
├── notebooks/
│   ├── EDA/                # Placeholder for exploratory data analysis notebooks
│   ├── model_tuning/       # Placeholder for hyperparameter tuning experiments
│   └── results_analysis/   # Placeholder for results visualization and analysis
├── results/
│   ├── logs/               # Placeholder for training and evaluation logs
│   ├── models/             # Placeholder for saved models
│   └── reports/            # Placeholder for performance reports
├── scripts/
│   ├── Flower_Client.py    # Federated client implementation
│   └── Input_prep.py       # Data preparation script
├── requirements.txt        # Placeholder for Python packages list
└── README.md               # Placeholder for project documentation

