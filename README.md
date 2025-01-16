# Repositório de Experimentos de Predição com Inteligência Artificial

## Descrição

Este repositório contém experimentos e testes para predição utilizando técnicas de inteligência artificial. O objetivo é explorar e avaliar diferentes abordagens para melhorar a acurácia e a eficiência dos modelos preditivos.

## Estrutura do Repositório

- **/data**: Contém conjuntos de dados utilizados nos experimentos.
- **/scripts**: Scripts para treinamento e avaliação dos modelos.
- **/utils**: Scripts para a montagem e pré-processamento dos conjuntos de dados de treinamento. Esses scripts requerem conexão com o banco de dados.
- **/results**: Resultados e métricas dos experimentos realizados.

## Requisitos

- Python 3.12
- Bibliotecas:
  - `certifi==2024.8.30`
  - `charset-normalizer==3.4.0`
  - `docopt==0.6.2`
  - `idna==3.10`
  - `requests==2.32.3`
  - `urllib3==2.2.3`
  - `yarg==0.1.10`
  - `python-dotenv`
  - `mysql-connector-python`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `xgboost`

## Licença
Este projeto está licenciado sob a Licença MIT.

## Ambiente

A conexão com o banco de dados é necessária somente para rodar os scripts de pré-processamento e montagem dos conjuntos de dados de treinamento que estão na pasta /utils.
Esses scripts utilizam dados armazenados no banco de dados MySQL e criam os conjuntos de dados que serão utilizados nos experimentos de predição.
Importante: Se você não precisar rodar os scripts de montagem dos conjuntos de dados, a conexão com o banco de dados não será necessária. O restante do projeto pode ser executado normalmente, sem necessidade de configurar o banco de dados.

## Como Configurar o Projeto

### 1. Instalar as Dependências

Para instalar todas as dependências necessárias para rodar o projeto, basta executar o comando `pip install -r requirements.txt`.

### 2. Configurar o Banco de Dados

O banco de dados MySQL utilizado no projeto está descrito no arquivo `utils/dataset/data-set.sql`. Para configurar o banco de dados:

1. Importe o arquivo `data-set.sql` no seu servidor MySQL com o comando a seguir: `mysql -u <DB_USER> -p <DB_DATABASE> < utils/dataset/data-set.sql`

    **Substitua** `<DB_USER>` pelo nome do usuário do MySQL e `<DB_DATABASE>` pelo nome do banco de dados onde você deseja importar o dataset.

2. Configure as credenciais de acesso no arquivo `.env` com as informações adequadas para o seu banco de dados.

