import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_dados(treino_path, teste_path):
    treino_df = pd.read_csv(treino_path)
    teste_df = pd.read_csv(teste_path)

    # Create a 'victory_casa' column as the target variable
    treino_df['victory_casa'] = treino_df['placar_casa'] > treino_df['placar_visitante']
    teste_df['victory_casa'] = teste_df['placar_casa'] > teste_df['placar_visitante']

    # Define input and output variables
    x_train = treino_df.drop(['victory_casa'], axis=1)
    y_train = treino_df['victory_casa']

    x_test = teste_df.drop(['victory_casa'], axis=1)
    y_test = teste_df['victory_casa']

    # Keep only numeric columns
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns
    x_train = x_train[numeric_features]
    x_test = x_test[numeric_features]

    # Create Min-Max scaler // Z-score wasn't working
    scaler = MinMaxScaler()

    # Normalize
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

def save_results_csv(path, results):
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    results_df.to_csv(path, index=False)

    print(f'Results saved to {path}')
