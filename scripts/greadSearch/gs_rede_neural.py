import os
import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def run_grid_search(treino_path, teste_path):
    # Carregar os dados
    treino_df = pd.read_csv(treino_path)
    teste_df = pd.read_csv(teste_path)

    # Criar a coluna 'victory_casa' como variável alvo
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

    # Normalizar os dados usando Min-Max Scaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definir os hiperparâmetros para o Grid Search
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }

    # Criar o MLPClassifier
    mlp = MLPClassifier(max_iter=1000, random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)

    # Obter os melhores hiperparâmetros
    best_params = grid_search.best_params_
    print(f'Melhores hiperparâmetros: {best_params}')

    # Avaliar o modelo com os melhores hiperparâmetros no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do melhor modelo: {accuracy:.2f}')

    return best_model, accuracy



temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"
]

porcentagens_treino = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # caminho atual voltando 2 pastas

# Dicionário para armazenar acurácias por porcentagem
accuracies_dict = {porcentagem: [] for porcentagem in porcentagens_treino}


# Início do temporizador
start_time = time.time()

for porcentagem in porcentagens_treino:
    for temporada in temporadas:
        porcentagem_str = str(porcentagem)

        # Construir os caminhos de treino e teste usando os.path
        treino_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "treino.csv")
        teste_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "teste.csv")

        # Executar o modelo e armazenar a acurácia
        print(f"Rodando modelo para temporada {temporada} e porcentagem {porcentagem_str}")
        best_model, accuracy = run_grid_search(treino_path, teste_path)
        accuracies_dict[porcentagem].append(accuracy)  # Adicionar a acurácia à lista correspondente

        break
    break

# Fim do temporizador
end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")