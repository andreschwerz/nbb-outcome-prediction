from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_svm(X_train, y_train):
    # Definir os hiperparâmetros para o Grid Search
    param_grid = {
        'C': [0.1, 1, 10],  # Regularização
        'kernel': ['linear', 'rbf', 'poly'],  # Tipos de kernel
        'gamma': ['scale', 'auto'],  # Coeficiente para kernels não lineares
        'degree': [3, 4]  # Para o kernel 'poly'
    }

    # Criar o modelo SVM
    svm = SVC(random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=2, scoring='accuracy', verbose=2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_svm(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        # Obter os melhores hiperparâmetros usando Grid Search
        best_params = get_hyper_params_svm(X_train, y_train)

        # Criar e treinar o modelo com os melhores hiperparâmetros
        model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            degree=best_params['degree'] if 'degree' in best_params else 3,
            random_state=42
        )
    else:
        # Modelo com hiperparâmetros padrão
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        best_params = []

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Para lidar com desbalanceamento de classes

    return accuracy, f1, best_params
