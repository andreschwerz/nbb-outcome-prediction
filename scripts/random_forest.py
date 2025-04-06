from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_rf(x_train, y_train):
    # Define hyperparameters for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 50],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]  # Sampling method
    }

    # Create the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Set up Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2)

    # Run Grid Search on the training set
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_rf(train_path, test_path, use_grid_search=True):
    x_train, x_test, y_train, y_test = read_dados(train_path, test_path)

    if use_grid_search:
        # Get the best hyperparameters using Grid Search
        best_params = get_hyper_params_rf(x_train, y_train)

        # Create and train the model with the best hyperparameters
        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap'],
            random_state=42
        )
    else:
        # Model with default hyperparameters
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_params = []

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Handles class imbalance

    return accuracy, f1, best_params