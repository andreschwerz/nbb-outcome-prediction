import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os

# Diretório onde estão os arquivos de treino e teste
# data_dir = '/home/alunos/a2252805/Área de Trabalho/experimentos-predi-o-nbb/data/experimento_02/2008-2009'
data_dir = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/2-1/'

# treino_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/treino.csv'
# teste_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/teste.csv'

# Número de arquivos de treino e teste (N)
N = 79  # Definir o número correto de arquivos de treino e teste

# Lista para armazenar as acurácias
accuracies = []

# Loop pelos arquivos de treino e teste
for i in range(1, N+1):
    treino_file = os.path.join(data_dir, f'treino_{i}.csv')
    teste_file = os.path.join(data_dir, f'teste_{i}.csv')

    # Carregar os dados
    treino_df = pd.read_csv(treino_file)
    teste_df = pd.read_csv(teste_file)

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
    print(f'Arquivo {i}: Acurácia do modelo: {accuracy:.2f}')

# Calcular a acurácia média de todos os modelos
mean_accuracy = sum(accuracies) / len(accuracies)
print(f'Acurácia média de todos os modelos: {mean_accuracy:.2f}')