import pandas as pd
from dados import get_jogos_temporada  # Importa a função do arquivo dados.py
import json
from sklearn.preprocessing import MinMaxScaler

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def descompactar_estatisticas(jogos):
    dados_formatados = []
    for jogo in jogos:
        casa_estatisticas = json.loads(jogo['estatisticas_casa'])
        visitante_estatisticas = json.loads(jogo['estatisticas_visitantes'])
        
        # Criar um dicionário com as estatísticas descompactadas
        dados = {
            'placar_casa': jogo['placar_casa'],
            'placar_visitante': jogo['placar_visitante'],
            'data': jogo['data'],
            'round': jogo['round'],
            'stage': jogo['stage'],
            'ano': jogo['ano'],
            'equipe_casa': jogo['equipe_casa'],
            'equipe_visitante': jogo['equipe_visitante'],
            
            # Estatísticas da equipe da casa
            'Pts_casa': casa_estatisticas.get('Pts', 0),
            '3P_casa': casa_estatisticas.get('3P', 0),
            '2P_casa': casa_estatisticas.get('2P', 0),
            'LL_casa': casa_estatisticas.get('LL', 0),
            'RT_casa': casa_estatisticas.get('RT', 0),
            'RO_casa': casa_estatisticas.get('RO', 0),
            'RD_casa': casa_estatisticas.get('RD', 0),
            'AS_casa': casa_estatisticas.get('AS', 0),
            'ER_casa': casa_estatisticas.get('ER', 0),
            'IA%_casa': casa_estatisticas.get('IA%', 0),
            '3PC_casa': casa_estatisticas.get('3PC', 0),
            '3PT_casa': casa_estatisticas.get('3PT', 0),
            '3P%_casa': casa_estatisticas.get('3P%', 0),
            '2PC_casa': casa_estatisticas.get('2PC', 0),
            '2PT_casa': casa_estatisticas.get('2PT', 0),
            '2P%_casa': casa_estatisticas.get('2P%', 0),
            'LLC_casa': casa_estatisticas.get('LLC', 0),
            'LLT_casa': casa_estatisticas.get('LLT', 0),
            'LL%_casa': casa_estatisticas.get('LL%', 0),
            'EN_casa': casa_estatisticas.get('EN', 0),
            'BR_casa': casa_estatisticas.get('BR', 0),
            'B/E_casa': casa_estatisticas.get('B/E', 0),
            'TO_casa': casa_estatisticas.get('TO', 0),
            'FC_casa': casa_estatisticas.get('FC', 0),
            'T/FC_casa': casa_estatisticas.get('T/FC', 0),
            'ET_casa': casa_estatisticas.get('ET', 0),
            'VI_casa': casa_estatisticas.get('VI', 0),
            'EF_casa': casa_estatisticas.get('EF', 0),
            
            # Estatísticas da equipe visitante
            'Pts_visitante': visitante_estatisticas.get('Pts', 0),
            '3P_visitante': visitante_estatisticas.get('3P', 0),
            '2P_visitante': visitante_estatisticas.get('2P', 0),
            'LL_visitante': visitante_estatisticas.get('LL', 0),
            'RT_visitante': visitante_estatisticas.get('RT', 0),
            'RO_visitante': visitante_estatisticas.get('RO', 0),
            'RD_visitante': visitante_estatisticas.get('RD', 0),
            'AS_visitante': visitante_estatisticas.get('AS', 0),
            'ER_visitante': visitante_estatisticas.get('ER', 0),
            'IA%_visitante': visitante_estatisticas.get('IA%', 0),
            '3PC_visitante': visitante_estatisticas.get('3PC', 0),
            '3PT_visitante': visitante_estatisticas.get('3PT', 0),
            '3P%_visitante': visitante_estatisticas.get('3P%', 0),
            '2PC_visitante': visitante_estatisticas.get('2PC', 0),
            '2PT_visitante': visitante_estatisticas.get('2PT', 0),
            '2P%_visitante': visitante_estatisticas.get('2P%', 0),
            'LLC_visitante': visitante_estatisticas.get('LLC', 0),
            'LLT_visitante': visitante_estatisticas.get('LLT', 0),
            'LL%_visitante': visitante_estatisticas.get('LL%', 0),
            'EN_visitante': visitante_estatisticas.get('EN', 0),
            'BR_visitante': visitante_estatisticas.get('BR', 0),
            'B/E_visitante': visitante_estatisticas.get('B/E', 0),
            'TO_visitante': visitante_estatisticas.get('TO', 0),
            'FC_visitante': visitante_estatisticas.get('FC', 0),
            'T/FC_visitante': visitante_estatisticas.get('T/FC', 0),
            'ET_visitante': visitante_estatisticas.get('ET', 0),
            'VI_visitante': visitante_estatisticas.get('VI', 0),
            'EF_visitante': visitante_estatisticas.get('EF', 0)
        }
        dados_formatados.append(dados)
    return dados_formatados

def normalizar_dados(dados):
    df = pd.DataFrame(dados)
    
    # Selecionar as colunas a serem normalizadas
    colunas_para_normalizar = [
        'placar_casa', 'placar_visitante',
        'Pts_casa', '3P_casa', '2P_casa', 'LL_casa', 'RT_casa', 'RO_casa', 'RD_casa', 'AS_casa', 
        'ER_casa', 'IA%_casa', '3PC_casa', '3PT_casa', '3P%_casa', '2PC_casa', '2PT_casa', 
        '2P%_casa', 'LLC_casa', 'LLT_casa', 'LL%_casa', 'EN_casa', 'BR_casa', 'B/E_casa', 'TO_casa', 
        'FC_casa', 'T/FC_casa', 'ET_casa', 'VI_casa', 'EF_casa', 'Pts_visitante', '3P_visitante', 
        '2P_visitante', 'LL_visitante', 'RT_visitante', 'RO_visitante', 'RD_visitante', 'AS_visitante', 
        'ER_visitante', 'IA%_visitante', '3PC_visitante', '3PT_visitante', '3P%_visitante', 
        '2PC_visitante', '2PT_visitante', '2P%_visitante', 'LLC_visitante', 'LLT_visitante', 
        'LL%_visitante', 'EN_visitante', 'BR_visitante', 'B/E_visitante', 'TO_visitante', 
        'FC_visitante', 'T/FC_visitante', 'ET_visitante', 'VI_visitante', 'EF_visitante'
    ]
    
    # Normalizar as colunas selecionadas
    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])
    
    return df.to_dict(orient='records')

def split_and_save_data():
    # Obter todos os jogos da temporada
    jogos = get_jogos_temporada('2008-2009')
    
    # Descompactar estatísticas
    jogos_formatados = descompactar_estatisticas(jogos)
    
    # # Normalizar os dados
    # jogos_normalizados = normalizar_dados(jogos_formatados)
    
    # Ordenar os jogos pela data
    jogos_sorted = sorted(jogos_formatados, key=lambda x: x['data'])
    
    # Definir o ponto de divisão (80% treino e 20% teste)
    split_index = int(len(jogos_sorted) * 0.4)
    
    # Dividir os dados em treino e teste com base na ordem das datas
    treino = jogos_sorted[:split_index]
    teste = jogos_sorted[split_index:]
    
    # Salvar os conjuntos em arquivos CSV
    save_to_csv(treino, 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/treino.csv')
    save_to_csv(teste, 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/teste.csv')

# Executar a divisão e salvar os arquivos
if __name__ == "__main__":
    split_and_save_data()
    print("Arquivos treino.csv e teste.csv foram criados com sucesso.")