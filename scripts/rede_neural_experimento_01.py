import os
import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


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
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model(treino_path, teste_path):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    # best_params = get_hyper_params_rede_neural(X_train, y_train)    

    # # Criar e treinar o modelo com os melhores hiperparâmetros
    # model = MLPClassifier(
    #     hidden_layer_sizes=best_params['hidden_layer_sizes'],
    #     activation=best_params['activation'],
    #     solver=best_params['solver'],
    #     learning_rate=best_params['learning_rate'],
    #     max_iter=10000,
    #     random_state=42
    # )

    model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=10000, random_state=42)

    # Treinar modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_results_csv(results):
    # Criar um DataFrame a partir dos resultados
    results_df = pd.DataFrame(results)

    # Salvar os resultados em um arquivo CSV
    output_path = os.path.join(base_path, 'results', 'rede_neural_experimento_01.csv')
    results_df.to_csv(output_path, index=False)

    print(f'Resultados salvos em {output_path}')


temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"
]
porcentagens_treino = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Lista para armazenar resultados
results = []

# Início do temporizador
start_time = time.time()

for porcentagem in porcentagens_treino:
    acuracias_por_porcentagem = []  # Para calcular a média e o desvio padrão dessa porcentagem

    for temporada in temporadas:
        porcentagem_str = str(porcentagem)

        # Construir os caminhos de treino e teste usando os.path
        treino_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "treino.csv")
        teste_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "teste.csv")

        # Executar o modelo e armazenar a acurácia
        print(f"Rodando modelo para temporada {temporada} e porcentagem {porcentagem_str}")
        accuracy = run_model(treino_path, teste_path)
        acuracias_por_porcentagem.append(accuracy)

        # Adicionar os resultados ao DataFrame
        results.append({
            'Temporada': temporada,
            'Porcentagem de Treino': porcentagem,
            'Acurácia': accuracy
        })

    # Calcular e adicionar a média e o desvio padrão ao final de cada porcentagem
    mean_accuracy = np.mean(acuracias_por_porcentagem)
    std_dev = np.std(acuracias_por_porcentagem)
    results.append({
        'Temporada': 'Média/Desvio Padrão',
        'Porcentagem de Treino': porcentagem,
        'Acurácia': f'Média: {mean_accuracy:.2f}, Desvio Padrão: {std_dev:.2f}'
    })

# Criar um DataFrame a partir dos resultados
results_df = pd.DataFrame(results)

# Salvar os resultados em um arquivo CSV
output_path = os.path.join(base_path, 'resultados_acuracias.csv')
results_df.to_csv(output_path, index=False)

print(f'Resultados salvos em {output_path}')

# Fim do temporizador
end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")