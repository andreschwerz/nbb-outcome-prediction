import os
import time
import pandas as pd
import numpy as np

from rede_neural import run_model_rede_neural
from experimentos import save_results_csv

from vanilla import run_model_vanilla

# vanilla, rede_neural
modelo = 'rede_neural'

temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"
]

porcentagens_treino = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
# porcentagens_treino = [0.2]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Lista para armazenar resultados
results = []
bests_params = []

# Início do temporizador
start_time = time.time()

for porcentagem in porcentagens_treino:
    acuracias_por_porcentagem = []  # Para calcular a média e o desvio padrão dessa porcentagem
    f1_scores_por_porcentagem = []

    for temporada in temporadas:
        porcentagem_str = str(porcentagem)

        # Construir os caminhos de treino e teste usando os.path
        treino_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "treino.csv")
        teste_path = os.path.join(base_path, "data", "experimento_01", temporada, porcentagem_str, "teste.csv")

        # Executar o modelo e armazenar a acurácia
        print(f"Rodando modelo para temporada {temporada} e porcentagem {porcentagem_str}")
        if(modelo == 'rede_neural'):
            accuracy, f1, best_params = run_model_rede_neural(treino_path, teste_path, True)
        elif(modelo == 'vanilla'):
            accuracy, f1, best_params = run_model_vanilla(treino_path, teste_path)

        bests_params.append(best_params)

        acuracias_por_porcentagem.append(accuracy)
        f1_scores_por_porcentagem.append(f1)

    # Calcular e adicionar a média e o desvio padrão ao final de cada porcentagem
    media_acuracia = np.mean(acuracias_por_porcentagem)
    media_f1_score = np.mean(f1_scores_por_porcentagem)
    
    desvio_padrao_acuracia = np.std(acuracias_por_porcentagem)
    desvio_padrao_f1_score = np.std(f1_scores_por_porcentagem)

    results.append({
        'Porcentagem de Treino': porcentagem,
        'Acurácia': f'{media_acuracia:.2f}',
        'Desvio Padrão Acurácia': f'{desvio_padrao_acuracia:.2f}',
        'F1-Score': f'{media_f1_score:.2f}',
        'Desvio Padrão F1-Score': f'{desvio_padrao_f1_score:.2f}',
    })

    path = os.path.join(base_path, 'results', 'experimento_01', f'{modelo}_experimento_01.csv')
    save_results_csv(path, results)

# Fim do temporizador
end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")

# Converter a lista de dicionários em um DataFrame
df_bests_params = pd.DataFrame(bests_params)

# Visualizar as primeiras linhas do DataFrame
print(df_bests_params)