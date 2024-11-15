import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import os
import time
import re

def read_dados(treino_path, teste_path):
    treino_df = pd.read_csv(treino_path)
    teste_df = pd.read_csv(teste_path)

    # Criar uma coluna 'victory_casa' como variável alvo
    treino_df['victory_casa'] = treino_df['placar_casa'] > treino_df['placar_visitante']
    teste_df['victory_casa'] = teste_df['placar_casa'] > teste_df['placar_visitante']

    # Definir variáveis de entrada e saída
    X_train = treino_df.drop(['victory_casa'], axis=1)
    y_train = treino_df['victory_casa']

    X_test = teste_df.drop(['victory_casa'], axis=1)
    y_test = teste_df['victory_casa']

    # Manter apenas as colunas numéricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train = X_train[numeric_features]
    X_test = X_test[numeric_features]

    # Criar o escalador Min-Max // Zscore não estava funcionando
    scaler = MinMaxScaler()

    # Normalizar
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_hyper_params_rede_neural(X_train, y_train):
    # Definir os hiperparâmetros para o Grid Search
    param_grid = {
        'max_iter':[10000],
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }

    # Criar o MLPClassifier
    mlp = MLPClassifier(max_iter=10000, random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2, scoring='accuracy', verbose=2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def extract_number(file_name):
    # Extrai o número do nome do arquivo.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def run_model(treino_path, teste_path):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    best_params = get_hyper_params_rede_neural(X_train, y_train)    

    #Criar e treinar o modelo com os melhores hiperparâmetros
    model = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        solver=best_params['solver'],
        learning_rate=best_params['learning_rate'],
        max_iter=10000,
        random_state=42
    )

    # Criar e treinar o modelo
    # model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy    

def run_models(data_dir):
    # Obter a lista de arquivos no diretório de treino e teste
    files = os.listdir(data_dir)

    # Filtrar e ordenar os arquivos de treino e teste com base nos números extraídos
    treino_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    teste_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Garantir que o número de arquivos de treino e teste é o mesmo
    if len(treino_files) != len(teste_files):
        print("Número de arquivos de treino e teste não coincide.")
        return

    accuracies_from_temporada = []

    # Loop pelos arquivos de treino e teste
    for treino_file, teste_file in zip(treino_files, teste_files):
        
        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)

        accuracy = run_model(treino_path, teste_path)
        accuracies_from_temporada.append(accuracy)
    
    return accuracies_from_temporada

if __name__ == '__main__':
    # Início do temporizador
    start_time_all = time.time()

    # Array para guardar os resultados de cada temporada / janela
    results = []

    temporadas = ['2008-2009', '2009-2010', '2011-2012',
                  '2012-2013', '2013-2014', '2014-2015',
                  '2015-2016', '2016-2017', '2018-2019', '2019-2020',
                  '2020-2021', '2021-2022', '2022-2023',
                  '2023-2024'
                 ]
    
    numeros_jogos_treino = [8,16,32,64,128]
    numeros_jogos_teste = [1,2,3,4]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for numero_jogos_teste in numeros_jogos_teste:
        for numero_jogos_treino in numeros_jogos_treino:
            for temporada in temporadas:    
                data_dir = os.path.join(base_path, "data", "experimento_02", temporada, f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}')

                # Rodando modelos para uma temporada
                print(f'Rodando modelo para temporada {temporada}, com janela {numero_jogos_treino} - {numero_jogos_teste}')
                accuracies_from_temporada = run_models(data_dir)
                mean_accuracy_from_temporada = sum(accuracies_from_temporada) / len(accuracies_from_temporada)

                # Guardando dados médios dos modelos para as janelas daquela temporada
                results.append({
                    'Temporada': temporada,
                    'Janela Flutuante': f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}',
                    'Acurácia média de todas': mean_accuracy_from_temporada
                })
            
            # Calcular e adicionar a média e o desvio daquela janela para todas as temporadas
            mean_accuracy = np.mean(mean_accuracy_from_temporada)
            std_dev = np.std(mean_accuracy_from_temporada)
            results.append({
                'Temporada': 'Média/Desvio Padrão',
                'Janela Flutuante': f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}',
                'Acurácia': f'Média: {mean_accuracy:.2f}, Desvio Padrão: {std_dev:.2f}'
            })

            # Criar um DataFrame a partir dos resultados
            results_df = pd.DataFrame(results)

            # Salvar os resultados em um arquivo CSV
            output_dir = os.path.join(base_path, 'results', 'experimento_02_gridsearch')
            os.makedirs(output_dir, exist_ok=True)

            # Salvar os resultados em um arquivo CSV
            output_path = os.path.join(output_dir, f'{numero_jogos_treino}-{numero_jogos_teste}.csv')
            results_df.to_csv(output_path, index=False)
            results = []

    # Fim do temporizador
    end_time_all = time.time()
    print(f"Tempo total de execução: {end_time_all - start_time_all:.2f} segundos")



