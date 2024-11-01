import os
import pandas as pd
import numpy as np  # Importar NumPy para calcular o desvio padrão
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def run_model(treino_path, teste_path):
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
    print(f'Acurácia do modelo: {accuracy:.2f}')
    return accuracy  # Retornar a acurácia em vez de imprimir

temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"
]

porcentagens_treino = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Dicionário para armazenar acurácias por porcentagem
accuracies_dict = {porcentagem: [] for porcentagem in porcentagens_treino}

for porcentagem in porcentagens_treino:
    for temporada in temporadas:
        porcentagem_str = str(porcentagem)

        # Construir os caminhos de treino e teste usando os.path
        treino_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "treino.csv")
        teste_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "teste.csv")

        # Executar o modelo e armazenar a acurácia
        print(f"Rodando modelo para temporada {temporada} e porcentagem {porcentagem_str}")
        accuracy = run_model(treino_path, teste_path)
        accuracies_dict[porcentagem].append(accuracy)  # Adicionar a acurácia à lista correspondente

# Calcular e exibir o desvio padrão para cada porcentagem
for porcentagem, accuracies in accuracies_dict.items():
    std_dev = np.std(accuracies)
    mean_accuracy = np.mean(accuracies)  # Calcular a média das acurácias
    print(f'Porcentagem de treino: {porcentagem}, Média da acurácia: {mean_accuracy:.2f}, Desvio padrão da acurácia: {std_dev:.2f}')
