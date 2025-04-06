from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_rede_neural(x_train, y_train):
    # Define hyperparameters for Grid Search
    param_grid = {
        'max_iter': [10000],
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }

    # Create the MLPClassifier
    mlp = MLPClassifier(max_iter=10000, random_state=42)

    # Set up Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2)

    # Run Grid Search on the training set
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_rede_neural(train_path, test_path, use_grid_search=True):
    x_train, x_test, y_train, y_test = read_dados(train_path, test_path)

    if use_grid_search:
        best_params = get_hyper_params_rede_neural(x_train, y_train)

        # Create and train the model with the best hyperparameters
        model = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            solver=best_params['solver'],
            learning_rate=best_params['learning_rate'],
            max_iter=best_params['max_iter'],
            random_state=42
        )

    else:
        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=10000, random_state=42)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' handles class imbalance  
                                                       # Calculates the F1 Score for each class individually, but weights by the number of samples of each class

    return accuracy, f1, best_params