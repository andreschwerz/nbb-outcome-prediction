import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Ajustar os caminhos dos arquivos para Windows
treino_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_01/2008-2009/0,9/treino.csv'
teste_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_01/2008-2009/0,9/teste.csv'

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

# Criar o pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42))
])

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
