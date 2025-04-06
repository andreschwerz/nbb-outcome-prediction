"""
This script reads the results of machine learning experiments from different models using CSV files,
processes the data, stores it in a dictionary, then converts the dictionary into a pandas DataFrame,
formats the accuracy and F1-score results, and saves the resulting DataFrame into a new CSV file.
"""

import csv
import os
import pandas as pd

models = ["svm", "random_forest", "naive_bayes", "rede_neural", "vanilla", "xgboost"]
models_no_underline = ["svm", "random-forest", "naive-bayes", "rede-neural", "vanilla", "xgboost"]
num_matches = 15

data_dict = {}

# Read each model's CSV file
for key, model in enumerate(models):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, f"{model}_experiment_01_{num_matches}.csv")

    # Open the CSV file
    with open(file_path) as file:
        data_results = []

        # Read the CSV file data
        reader = csv.reader(file)
        for row in reader:
            # Process each row
            if row[1] == '-':
                data_results.append(row)

    data_dict[f"{models_no_underline[key]}_{num_matches}"] = data_results

print(data_dict)

def dict_to_dataframe(data_dict):
    # Create an empty dataframe
    df = pd.DataFrame()

    # Iterate over the dictionary items
    for key, values in data_dict.items():
        # Extract model and number of matches from the key
        model, num_matches = key.split("_")

        # Create a new dataframe for each model and match count
        new_df = pd.DataFrame(columns=[f"{model}_{num_matches}_accuracy", f"{model}_{num_matches}_f1score"])

        # Iterate again to fill in values
        for key, values in data_dict.items():
            model, num_matches = key.split("_")

            # Temporary dataframe with column names translated and in snake_case
            temp_df = pd.DataFrame(values, columns=["train_percentage", "season", "accuracy", "accuracy_std", "f1_score", "f1_score_std"])

            new_df[f"{model}_{num_matches}_accuracy"] = (
                temp_df['accuracy'].astype(str).str.replace('.', ',', regex=False) +
                ' ± ' + temp_df['accuracy_std'].astype(str).str.replace('.', ',', regex=False)
            )
            new_df[f"{model}_{num_matches}_f1score"] = (
                temp_df['f1_score'].astype(str).str.replace('.', ',', regex=False) +
                ' ± ' + temp_df['f1_score_std'].astype(str).str.replace('.', ',', regex=False)
            )

        return new_df

df = dict_to_dataframe(data_dict)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
file_path = os.path.join(base_path, f"resultados_experiment_01_{num_matches}.csv")

df.to_csv(file_path, index=False)
