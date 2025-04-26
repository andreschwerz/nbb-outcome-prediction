from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experiments import read_data

def get_hyper_params_xgboost(X_train, y_train):
    # Define the hyperparameters for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],  # Number of trees
        'max_depth': [3, 5, 7],         # Maximum depth of the tree
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'subsample': [0.8, 1.0],        # Fraction of samples for each tree
        'colsample_bytree': [0.8, 1.0], # Fraction of features for each tree
    }

    # Create the XGBoost model
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # Set up the Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2)

    try:
        # Run the Grid Search on the training set
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Error during Grid Search: {e}")
        return None

    return best_params

def run_model_xgboost(train_path, test_path, use_grid_search=True):
    X_train, X_test, y_train, y_test, feature_names_dict = read_data(train_path, test_path)

    if use_grid_search:
        # Get the best hyperparameters using Grid Search
        best_params = get_hyper_params_xgboost(X_train, y_train)

        if(best_params is None):
            return None, None, None

        # Create and train the model with the best hyperparameters
        model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            eval_metric='logloss',
            random_state=42
        )
    else:
        # Model with default hyperparameters
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric='logloss',
            random_state=42
        )
        best_params = []

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions with the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # To handle class imbalance

    return accuracy, f1, best_params, model, feature_names_dict
