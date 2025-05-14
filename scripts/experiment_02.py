import os
import time
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiments import save_results_csv
from xgboost_model import run_model_xgboost

model = 'xgb'
best_params_list = []
feature_importance_sum = {}

def extract_number(file_name):
    # Extract the number from the file name.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def run_models(data_dir):
    # Get the list of files in the train and test directory
    files = os.listdir(data_dir)

    # Filter and sort training and testing files based on the extracted numbers
    train_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    test_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Ensure the number of training and testing files matches
    if len(train_files) != len(test_files):
        print("Number of training and testing files does not match.")
        return

    accuracies_by_season = []
    f1_scores_by_season = []

    global feature_importance_sum  # <- add this to use the global dict

    # Loop through training and testing files
    for train_file, test_file in zip(train_files, test_files):
        train_path = os.path.join(data_dir, train_file)
        test_path = os.path.join(data_dir, test_file)

        accuracy, f1, best_params, model, feature_names_dict = run_model_xgboost(train_path, test_path, False)

        if accuracy is None:
            continue
        else:
            best_params_list.append(best_params)
            accuracies_by_season.append(accuracy)
            f1_scores_by_season.append(f1)

            # Getting feature importance
            booster = model.get_booster()
            feature_scores = booster.get_score(importance_type='gain')

            # Updating the importance accumulator
            for feature_name, importance in feature_scores.items():
                if feature_name in feature_importance_sum:
                    feature_importance_sum[feature_name] += importance
                else:
                    feature_importance_sum[feature_name] = importance
    
    # 🔥 After finishing all files for the season:
    if feature_importance_sum:
        # Creating the importance DataFrame
        importance_df = pd.DataFrame.from_dict(feature_importance_sum, orient='index', columns=['Importance'])

        # Mapping feature names f0, f1, etc., to the actual feature name
        def map_feature_name(f):
            # Extract the number after "f" and convert to int
            idx = int(f[1:])
            return feature_names_dict.get(idx, f)  # If not found in the dictionary, keep the original
        
        # Dicionário com descrições detalhadas das features
        feature_descriptions = {
            'placar_casa': 'Placar da equipe da casa',
            'placar_visitante': 'Placar da equipe visitante',
            'EF_casa': 'Eficiência ofensiva da casa',
            'RT_casa': 'Rebotes totais da casa',
            'AS_visitante': 'Assistências do visitante',
            'B/E_visitante': 'Bloqueios e erros do visitante',
            '2P%_casa': 'Aproveitamento de 2 pontos da casa (%)',
            'Pts_casa': 'Pontos da casa',
            '2P_casa': 'Cestas de 2 pontos da casa',
            'LL%_visitante': 'Aproveitamento de lances livres do visitante (%)',
            'EF_visitante': 'Eficiência ofensiva do visitante',
            '2PT_visitante': 'Cestas de 2 pontos do visitante',
            'RD_visitante': 'Rebotes defensivos do visitante',
            'EN_casa': 'Erros não forçados da casa',
            '3P%_casa': 'Aproveitamento de 3 pontos da casa (%)'
        }

        # Renaming the index
        importance_df.index = importance_df.index.map(map_feature_name)

        # Sorting by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Selecting only 15 main features
        top_features_df = importance_df.head(15)

        # Saving the global feature importance
        global_output_dir = os.path.join(base_path, 'results', 'experiment_02', 'feature_importance')
        os.makedirs(global_output_dir, exist_ok=True)
        
        # Plotando o gráfico
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(top_features_df.index, top_features_df['Importance'])

        # Título e eixos
        ax.set_title("Top 15 Global Feature Importance")
        ax.set_ylabel("Importance (sum across all models)")
        plt.xticks(rotation=45, ha='right')

        # Monta a legenda personalizada com os nomes reais das features
        #feature_legends = [f"{abbr}: {full}" for abbr, full in zip(top_features_df.index, top_features_df.index)]
        feature_legends = [f"{abbr}: {feature_descriptions.get(abbr, abbr)}" for abbr in top_features_df.index]

        # Junta tudo em uma string
        legend_text = "\n".join(feature_legends)

        # Estilo da caixa
        props = dict(facecolor='white', alpha=0.8)

        # Adiciona a legenda dentro da área do gráfico, por exemplo no canto superior direito
        ax.text(0.50, 0.98, legend_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(os.path.join(global_output_dir, 'global_feature_importance.png'))
        plt.close()

    avg_accuracy = np.mean(accuracies_by_season)
    avg_f1_score = np.mean(f1_scores_by_season)

    return avg_accuracy, avg_f1_score


if __name__ == '__main__':
    # Start the timer
    start_time_all = time.time()

    # Array to store the results of each season/window
    results = []

    seasons = [
        #'2008-2009', '2009-2010', '2011-2012',
        #'2012-2013', '2013-2014', '2014-2015',
        #'2015-2016', '2016-2017', '2018-2019', '2019-2020',
        #'2020-2021', '2021-2022', '2022-2023',
        '2023-2024'
    ]
    
    train_games_counts = [128]
    test_games_counts = [8]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for train_games_count in train_games_counts:
        for test_games_count in test_games_counts:

            f1_scores_all_seasons = []
            accuracies_all_seasons = []

            for season in seasons:
                data_dir = os.path.join(base_path, "data", "experiment_02", season, f'{train_games_count}-{test_games_count}')

                # Running models for one season
                print(f'Running model for season {season}, with window {train_games_count} - {test_games_count}')

                accuracy_by_season, f1_score_by_season = run_models(data_dir)
                
                accuracies_all_seasons.append(accuracy_by_season)
                f1_scores_all_seasons.append(f1_score_by_season)

                results.append({
                    'Sliding Window': f'{train_games_count}-{test_games_count}',
                    'Season': season,
                    'Accuracy': f'{accuracy_by_season:.2f}',
                    'Accuracy Std Dev': '-',
                    'F1-Score': f'{f1_score_by_season:.2f}',
                    'F1-Score Std Dev': '-',
                })

                output_dir = os.path.join(base_path, 'results', 'experiment_02')
                output_path = os.path.join(output_dir, f'{model}_experiment_02_train_{train_games_count}_partial.csv')
                save_results_csv(output_path, results)
            
            # Calculate and add the average and standard deviation for that window across all seasons
            avg_accuracy = np.mean(accuracies_all_seasons)
            avg_f1_score = np.mean(f1_scores_all_seasons)

            std_dev_accuracy = np.std(accuracies_all_seasons)
            std_dev_f1_score = np.std(f1_scores_all_seasons)

            results.append({
                'Sliding Window': f'{train_games_count}-{test_games_count}',
                'Season': '-',
                'Accuracy': f'{avg_accuracy:.2f}',
                'Accuracy Std Dev': f'{std_dev_accuracy:.2f}',
                'F1-Score': f'{avg_f1_score:.2f}',
                'F1-Score Std Dev': f'{std_dev_f1_score:.2f}',
            })

            output_dir = os.path.join(base_path, 'results', 'experiment_02')
            output_path = os.path.join(output_dir, f'{model}_experiment_02_train_{train_games_count}.csv')
            save_results_csv(output_path, results)

    # End the timer
    end_time_all = time.time()
    print(f"Total execution time: {end_time_all - start_time_all:.2f} seconds")
