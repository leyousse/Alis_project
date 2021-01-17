import mysql.connector
from mysql.connector import Error
import json

with open('config.json') as config_file:
    data = json.load(config_file)

host = data['database']['host']
user = data['database']['user']
password = data['database']['password']
db = data['database']['db']

def connect_to_databse():
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            db=db)
        if conn.is_connected():
            cur = conn.cursor()
            print("conected to database")
            

    except Error as e:
        print("Error while connecting to MySQL", e)
    return conn,cur

def close_connection(cur,conn):
    cur.close()
    conn.close()
    print("MySQL connection is closed")

def create_database_interlocuteur(cur,conn):
    cur.execute("""CREATE TABLE IF NOT EXISTS data_interlocuteur
                (
                    id int NOT NULL AUTO_INCREMENT,
                    phrase varchar(255),
                    interlocuteur varchar(100),
                    PRIMARY KEY (id)
                );""")
    conn.commit()

def create_database_perso(cur,conn):
    cur.execute("""CREATE TABLE IF NOT EXISTS data_perso 
                (
                    id int NOT NULL AUTO_INCREMENT,
                    phrase varchar(255),
                    interlocuteur varchar(100),
                    compteur int,
                    PRIMARY KEY (id)
                );""")
    conn.commit()

def insert(cur,conn,name_table,phrase,interlocuteur):
    cur.execute(f"""
                INSERT INTO {name_table}(phrase,interlocuteur) VALUES ('{phrase}','{interlocuteur}')
                """)
    conn.commit()

def parse_json(request):
    interlocuteur = request['interlocuteur']
    phrase = request['phrase']
    print(f"L'interlocuteur est {interlocuteur}, il a dit {phrase}")
    
    conn,cur = connect_to_databse()
    create_database_interlocuteur(cur,conn)
    insert(cur,conn,'data_perso',phrase,interlocuteur)
    close_connection(cur,conn)

## sur les top phrases le mec potentiellement il va pas dire 2 fois exactement la meme phrase
