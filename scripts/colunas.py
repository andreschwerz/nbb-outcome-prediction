import pandas as pd

treino_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/treino.csv'
teste_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/teste.csv'

treino_df = pd.read_csv(treino_path)
teste_df = pd.read_csv(teste_path)

# Exibir as colunas dos DataFrames
print("Colunas no treino_df:", treino_df.columns)
print("Colunas no teste_df:", teste_df.columns)
