from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experiments import read_data

def get_hyper_params_svm(x_train, y_train):
    # Define hyperparameters for Grid Search
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
        'gamma': ['scale', 'auto'],  # Coefficient for non-linear kernels
        'degree': [3, 4]  # For the 'poly' kernel
    }

    # Create SVM model
    svm = SVC(random_state=42)

    # Configure Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2)

    # Run Grid Search on the training set
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_svm(train_path, test_path, use_grid_search=True):
    x_train, x_test, y_train, y_test = read_data(train_path, test_path)

    if use_grid_search:
        # Get best hyperparameters using Grid Search
        best_params = get_hyper_params_svm(x_train, y_train)

        # Create and train model with the best hyperparameters
        model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            degree=best_params['degree'] if 'degree' in best_params else 3,
            random_state=42
        )
    else:
        # Model with default hyperparameters
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        best_params = []

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Handles class imbalance

    return accuracy, f1, best_params
