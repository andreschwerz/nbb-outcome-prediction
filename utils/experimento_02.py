import pandas as pd
from dados import get_jogos_temporada
from dados import formatar_medias

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def descompactar_estatisticas(jogos):
    dados_formatados = []

    for jogo in jogos:
        casa_estatisticas = jogo['estatisticas_casa']
        visitante_estatisticas = jogo['estatisticas_visitantes']
        
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
            # Estatísticas da equipe visitante
            'Pts_visitante': visitante_estatisticas.get('Pts', 0),
            '3P_visitante': visitante_estatisticas.get('3P', 0),
            '2P_visitante': visitante_estatisticas.get('2P', 0),
            'LL_visitante': visitante_estatisticas.get('LL', 0),
            'RT_visitante': visitante_estatisticas.get('RT', 0),
            'RO_visitante': visitante_estatisticas.get('RO', 0),
            'RD_visitante': visitante_estatisticas.get('RD', 0),
            'AS_visitante': visitante_estatisticas.get('AS', 0)
        }
        dados_formatados.append(dados)
    return dados_formatados

def gerar_arquivos_treino_teste(temporada, qtd_jogos_treino, qtd_jogos_teste):
    jogos_treino = get_jogos_temporada(temporada)
    jogos_teste = get_jogos_temporada(temporada)
    
    # Formatar os jogos com médias para treino
    jogos_treino_formatados = formatar_medias(jogos_treino, True)
    jogos_teste_formatados = formatar_medias(jogos_teste, False)

    indice = 0
    num_arquivo = 1
    
    while indice < len(jogos_treino_formatados):
        # Definir os blocos de treino e teste
        treino = jogos_treino_formatados[indice:indice+qtd_jogos_treino]
        teste = jogos_teste_formatados[indice+qtd_jogos_treino:indice+qtd_jogos_treino+qtd_jogos_teste]
        
        # Parar se não houver jogos suficientes para teste
        if not teste:
            break

        # Descompactar as estatísticas
        treino_formatado = descompactar_estatisticas(treino)
        teste_formatado = descompactar_estatisticas(teste)

        # Salvar os arquivos de treino e teste
        save_to_csv(treino_formatado, f'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/treino_{num_arquivo}.csv')
        save_to_csv(teste_formatado, f'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/teste_{num_arquivo}.csv')

        print(f'Arquivos treino_{num_arquivo}.csv e teste_{num_arquivo}.csv foram criados com sucesso.')

        # Incrementar para o próximo lote de jogos
        indice += qtd_jogos_treino + qtd_jogos_teste
        num_arquivo += 1

# Exemplo de uso
if __name__ == "__main__":
    temporada = '2008-2009'
    qtd_jogos_treino = 8
    qtd_jogos_teste = 1
    gerar_arquivos_treino_teste(temporada, qtd_jogos_treino, qtd_jogos_teste)
