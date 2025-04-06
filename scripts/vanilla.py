from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from experimentos import read_dados

def run_model_vanilla(train_path, test_path):
    # Reads the data (using a mock function to simulate the data)
    X_train, X_test, y_train, y_test = read_dados(train_path, test_path)

    # Simulates predictions: everything as True
    y_pred = np.ones_like(y_test)  # Creates an array with the same size as y_test and fills it with True (or 1)

    # Evaluates performance by simulating 100% of predictions as "True"
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Returns the simulated results
    return accuracy, f1, []