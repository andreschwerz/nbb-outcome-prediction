import pandas as pd
import os
from dados import get_jogos_temporada
from dados import formatar_medias

def save_to_csv(data, filepath):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Cria o diretório se não existir
    df.to_csv(filepath, index=False)

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

def gerar_arquivos_treino_teste(temporada, qtd_jogos_treino, qtd_jogos_teste, base_path):
    jogos_treino = get_jogos_temporada(temporada)
    jogos_teste = get_jogos_temporada(temporada)
    
    # Formatar os jogos com médias para treino
    jogos_treino_formatados = formatar_medias(jogos_treino, True)
    jogos_teste_formatados = formatar_medias(jogos_teste, False)

    indice = 0
    num_arquivo = 1
    
    while indice < len(jogos_treino_formatados):
        treino = jogos_treino_formatados[indice:indice+qtd_jogos_treino]
        teste = jogos_teste_formatados[indice+qtd_jogos_treino:indice+qtd_jogos_treino+qtd_jogos_teste]

        if not teste:
            break

        treino_formatado = descompactar_estatisticas(treino)
        teste_formatado = descompactar_estatisticas(teste)

        final_path = base_path + f'{temporada}' + '/' + f'{qtd_jogos_treino}' + '-' + f'{qtd_jogos_teste}' + '/'
        # Definir o caminho do diretório para salvar os arquivos
        temporada_path = os.path.join(final_path)
        
        # Salvar os arquivos de treino e teste
        save_to_csv(treino_formatado, f'{temporada_path}/treino_{num_arquivo}.csv')
        save_to_csv(teste_formatado, f'{temporada_path}/teste_{num_arquivo}.csv')

        print(f'Arquivos treino_{num_arquivo}.csv e teste_{num_arquivo}.csv foram criados com sucesso.')

        indice += qtd_jogos_treino + qtd_jogos_teste
        num_arquivo += 1

# Exemplo de uso
if __name__ == "__main__":
    temporada = '2008-2009'
    qtd_jogos_treino = 16
    qtd_jogos_teste = 1
    # base_path = 'C:/Users/rafae/OneDrive/Área de Trabalho/TCC/experimentos/experimentos-predi-o-nbb/data/experimento_02/'
    base_path = '/home/alunos/a2252805/Área de Trabalho/experimentos-predi-o-nbb/data/experimento_02/'

    gerar_arquivos_treino_teste(temporada, qtd_jogos_treino, qtd_jogos_teste, base_path)

