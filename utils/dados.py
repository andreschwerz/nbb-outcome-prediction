import mysql.connector

# Configuração de conexão com o banco de dados
db_config = {
    'user': 'root',          # Substitua com seu usuário
    'password': '',         # Substitua com sua senha
    'host': 'localhost',    # Ou o endereço do seu banco de dados
    'database': 'mydb'
}

def get_all_jogos():
    conn = mysql.connector.connect(**db_config)
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos ORDER BY data"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_casa(equipe):
    conn = mysql.connector.connect(**db_config)
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_casa = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_visitante(equipe):
    conn = mysql.connector.connect(**db_config)
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_visitante = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_temporada(ano):
    conn = mysql.connector.connect(**db_config)
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE ano = %s ORDER BY data"
        cursor.execute(query, (ano,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

# Exemplo de uso
if __name__ == "__main__":

    temporada = '2008-2009'
    jogos_temporada = get_jogos_temporada(temporada)
    
    print(f"Jogos da temporada {temporada}:")
    for jogo in jogos_temporada:
        print(jogo, "\n\n")
