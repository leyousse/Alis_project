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

def create_database(cur,conn,name_db):
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {name_db} 
                (
                    id int NOT NULL AUTO_INCREMENT,
                    phrase varchar(255),
                    interlocuteur varchar(100),
                    reponse varchar(255),
                    PRIMARY KEY (id)
                );""")
    conn.commit()

def insert(cur,conn,name_table,phrase,interlocuteur,reponse):
    cur.execute(f"""
                INSERT INTO {name_table}(phrase,interlocuteur,reponse) VALUES ('{phrase}','{interlocuteur}','{reponse}')
                """)
    conn.commit()

def insert_phrase(phrase,reponse,interlocuteur):
    status = 0
    try :
        NAME_DB = "mes_conversations"
        conn,cur = connect_to_databse()
        create_database(cur,conn,NAME_DB)
        insert(cur,conn,NAME_DB,phrase,interlocuteur,reponse)
        close_connection(cur,conn)
        status = 1
    except Error as e:
        print("Il y a eu un problème",e)
    return status
