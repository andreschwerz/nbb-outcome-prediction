import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import os
import re

def extract_number(file_name):
    # Extrai o número do nome do arquivo.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def run_model(data_dir, temporada):
    # Lista para armazenar as acurácias
    accuracies = []

    # Obter a lista de arquivos no diretório de treino e teste
    files = os.listdir(data_dir)

    # Filtrar e ordenar os arquivos de treino e teste com base nos números extraídos
    treino_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    teste_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Garantir que o número de arquivos de treino e teste é o mesmo
    if len(treino_files) != len(teste_files):
        print("Número de arquivos de treino e teste não coincide.")
        return

    # Loop pelos arquivos de treino e teste
    for treino_file, teste_file in zip(treino_files, teste_files):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)

        # Carregar os dados
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

        # Criar o escalador Min-Max
        scaler = MinMaxScaler()
        # Normalizar
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Criar e treinar o modelo
        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Fazer previsões com o conjunto de teste
        y_pred = model.predict(X_test)

        # Avaliar o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)  # Armazenar a acurácia na lista
        # print(f'Arquivo {treino_file}: Acurácia do modelo: {accuracy:.2f}')

    # Calcular a acurácia média de todos os modelos
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'Acurácia média de todos os modelos - {temporada}: {mean_accuracy:.2f}')

if __name__ == '__main__':
    temporadas = ['2008-2009', '2009-2010', '2011-2012',
                  '2012-2013', '2013-2014', '2014-2015',
                  '2015-2016', '2016-2017', '2018-2019', '2019-2020',
                  '2020-2021', '2021-2022', '2022-2023',
                  '2023-2024'
                 ]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for temporada in temporadas:
        data_dir = f'{base_path}/data/experimento_02/{temporada}/8-1/'
        run_model(data_dir, temporada)

