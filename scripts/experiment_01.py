import os
import time
import pandas as pd
import numpy as np

from neural_network import run_model_neural_network
from vanilla import run_model_vanilla
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from xgboost_model import run_model_xgboost

from experiments import save_results_csv

# vanilla, neural_network, svm, random_forest, naive_bayes, xgboost
model = 'xgboost'
games_averages = ['5', '10', '15']

seasons = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"
]

training_percentages = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# List to store results
results = []
best_params_list = []

# Start the timer
start_time = time.time()

for games_average in games_averages:
    for percentage in training_percentages:
        accuracies_by_percentage = []
        f1_scores_by_percentage = []

        for season in seasons:
            percentage_str = str(percentage)

            # Build training and test paths using os.path
            train_path = os.path.join(base_path, "data", "experiment_01", games_average, season, percentage_str, "treino.csv")
            test_path = os.path.join(base_path, "data", "experiment_01", games_average, season, percentage_str, "teste.csv")

            # Run the model and store the accuracy
            print(f"Running model for season {season} and percentage {percentage_str}")

            if model == 'neural_network':
                accuracy, f1, best_params = run_model_neural_network(train_path, test_path, True)

            elif model == 'vanilla':
                accuracy, f1, best_params = run_model_vanilla(train_path, test_path)

            elif model == 'svm':
                accuracy, f1, best_params = run_model_svm(train_path, test_path, True)

            elif model == 'random_forest':
                accuracy, f1, best_params = run_model_rf(train_path, test_path, True)

            elif model == 'naive_bayes':
                accuracy, f1, best_params = run_model_naive_bayes(train_path, test_path)

            elif model == 'xgboost':
                accuracy, f1, best_params = run_model_xgboost(train_path, test_path, True)

            accuracies_by_percentage.append(accuracy)
            f1_scores_by_percentage.append(f1)

            results.append({
                'Training Percentage': percentage,
                'Season': season,
                'Accuracy': f'{accuracy:.2f}',
                'Accuracy Std Dev': '-',
                'F1-Score': f'{f1:.2f}',
                'F1-Score Std Dev': '-',
            })

        # Calculate and add average and standard deviation at the end of each percentage
        avg_accuracy = np.mean(accuracies_by_percentage)
        avg_f1_score = np.mean(f1_scores_by_percentage)
        
        std_dev_accuracy = np.std(accuracies_by_percentage)
        std_dev_f1_score = np.std(f1_scores_by_percentage)

        results.append({
            'Training Percentage': percentage,
            'Season': '-',
            'Accuracy': f'{avg_accuracy:.2f}',
            'Accuracy Std Dev': f'{std_dev_accuracy:.2f}',
            'F1-Score': f'{avg_f1_score:.2f}',
            'F1-Score Std Dev': f'{std_dev_f1_score:.2f}',
        })

        path = os.path.join(base_path, 'results', 'experiment_01', f'{model}_experiment_01_{games_average}.csv')
        save_results_csv(path, results)

    results = []

# End the timer
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
