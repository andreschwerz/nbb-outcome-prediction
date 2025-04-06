from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def run_model_naive_bayes(train_path, test_path, use_grid_search=False):
    x_train, x_test, y_train, y_test = read_dados(train_path, test_path)

    # Create the Naïve Bayes model
    model = GaussianNB()

    # Naïve Bayes does not use Grid Search for hyperparameter tuning,
    # but we keep the `use_grid_search` parameter for consistency with other classifiers.
    best_params = []  # No tunable hyperparameters here

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Handles class imbalance

    return accuracy, f1, best_params
