import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Create a 'victory_home' column as the target variable
    train_df['victory_home'] = train_df['placar_casa'] > train_df['placar_visitante']
    test_df['victory_home'] = test_df['placar_casa'] > test_df['placar_visitante']

    # Define input and output variables
    x_train = train_df.drop(['victory_home'], axis=1)
    y_train = train_df['victory_home']

    x_test = test_df.drop(['victory_home'], axis=1)
    y_test = test_df['victory_home']

    # Keep only numeric columns
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns
    x_train = x_train[numeric_features]
    x_test = x_test[numeric_features]

    # Create Min-Max scaler // Z-score wasn't working
    scaler = MinMaxScaler()

    # Normalize
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create a dictionary for feature names
    feature_names_dict = {i: name for i, name in enumerate(numeric_features)}

    return x_train, x_test, y_train, y_test, feature_names_dict

def save_results_csv(path, results):
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    results_df.to_csv(path, index=False)

    print(f'Results saved to {path}')
