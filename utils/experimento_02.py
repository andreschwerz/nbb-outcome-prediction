import pandas as pd
from dados import get_jogos_temporada
from dados import formatar_medias

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def descompactar_estatisticas(jogos):
    dados_formatados = []

    for jogo in jogos:
        print('\n', jogo)

    for jogo in jogos:
        casa_estatisticas = jogo['estatisticas_casa']  # já é um dicionário
        visitante_estatisticas = jogo['estatisticas_visitantes']  # já é um dicionário
        
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
        # save_to_csv(treino_formatado, f'/home/alunos/a2252805/Área de Trabalho/experimentos-predi-o-nbb/data/experimento_02/2008-2009/treino_{num_arquivo}.csv')
        # save_to_csv(teste_formatado, f'/home/alunos/a2252805/Área de Trabalho/experimentos-predi-o-nbb/data/experimento_02/2008-2009/teste_{num_arquivo}.csv')


        save_to_csv(treino_formatado, f'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/2-1/treino_{num_arquivo}.csv')
        save_to_csv(teste_formatado, f'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/2008-2009/2-1/teste_{num_arquivo}.csv')

        print(f'Arquivos treino_{num_arquivo}.csv e teste_{num_arquivo}.csv foram criados com sucesso.')

        # Incrementar para o próximo lote de jogos
        indice += qtd_jogos_treino + qtd_jogos_teste
        num_arquivo += 1

# Exemplo de uso
if __name__ == "__main__":
    temporada = '2008-2009'
    qtd_jogos_treino = 2
    qtd_jogos_teste = 1
    gerar_arquivos_treino_teste(temporada, qtd_jogos_treino, qtd_jogos_teste)
