# Artificial Intelligence Prediction Experiments Repository

## Description

This repository contains experiments and tests for prediction using artificial intelligence techniques. The goal is to explore and evaluate different approaches to improve the accuracy and efficiency of predictive models.

## Repository Structure

- **/data**: Contains datasets used in the experiments.
- **/scripts**: Scripts for training and evaluating the models.
- **/utils**: Scripts for assembling and pre-processing the training datasets. These scripts require a connection to the database.
- **/results**: Results and metrics of the experiments conducted.

## Requirements

- Python 3.12
- Libraries:
  - `certifi==2024.8.30`
  - `charset-normalizer==3.4.0`
  - `docopt==0.6.2`
  - `idna==3.10`
  - `requests==2.32.3`
  - `urllib3==2.2.3`
  - `yarg==0.1.10`
  - `python-dotenv`
  - `mysql-connector-python`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `xgboost`

## License
This project is licensed under the MIT License.

## Environment

A connection to the database is only required to run the pre-processing and dataset assembly scripts located in the /utils folder. These scripts use data stored in the MySQL database and create the datasets to be used in the prediction experiments.  
**Important**: If you do not need to run the dataset assembly scripts, the database connection will not be necessary. The rest of the project can be run normally without the need to set up the database.

## How to Set Up the Project

### 1. Install Dependencies

To install all the required dependencies to run the project, simply execute the command: `pip install -r requirements.txt`.

### 2. Configure the Database

The MySQL database used in the project is described in the `utils/dataset/data-set.sql` file. To set up the database:

1. Import the  `data-set.sql` file into your MySQL server using the following command: `mysql -u <DB_USER> -p <DB_DATABASE> < utils/dataset/data-set.sql`

    **Replace** `<DB_USER>` with the MySQL username and `<DB_DATABASE>` with the name of the database where you want to import the dataset.

2. Configure the access credentials in the `.env` file with the appropriate information for your database.