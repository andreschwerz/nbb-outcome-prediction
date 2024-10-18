import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os
import re

def extract_number(file_name):
    """Extrai o número do nome do arquivo."""
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def run_model(data_dir):
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

        # Criar a coluna 'victory_casa' como variável alvo
        treino_df['victory_casa'] = treino_df['placar_casa'] > treino_df['placar_visitante']
        teste_df['victory_casa'] = teste_df['placar_casa'] > teste_df['placar_visitante']

        # Definir variáveis de entrada e saída
        X_train = treino_df.drop(['victory_casa'], axis=1)
        y_train = treino_df['victory_casa']

        X_test = teste_df.drop(['victory_casa'], axis=1)
        y_test = teste_df['victory_casa']

        # Separar colunas numéricas e categóricas
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        # Pré-processamento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Preencher valores ausentes
                    ('scaler', StandardScaler())  # Normalizar dados numéricos
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Preencher valores ausentes
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificar variáveis categóricas
                ]), categorical_features)
            ])

        # Criar o pipeline do modelo
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42))
        ])

        # Treinar o modelo com o conjunto de treino
        model.fit(X_train, y_train)

        # Fazer previsões com o conjunto de teste
        y_pred = model.predict(X_test)

        # Avaliar o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)  # Armazenar a acurácia na lista
        print(f'Arquivo {treino_file}: Acurácia do modelo: {accuracy:.2f}')

    # Calcular a acurácia média de todos os modelos
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'Acurácia média de todos os modelos: {mean_accuracy:.2f}')

if __name__ == '__main__':
    data_dir = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/8-1/'
    # data_dir = '/home/alunos/a2252805/Área de Trabalho/experimentos-predi-o-nbb/data/experimento_02/all/8-1/'

    run_model(data_dir)


