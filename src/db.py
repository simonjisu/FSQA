import psycopg
from psycopg import sql

class DBHandler():
    def __init__(self, db_settings):
        self.db_settings = db_settings

    def query(self, query):
        with psycopg.connect(
            dbname=self.db_settings['dbname'],
            host=self.db_settings['host'],
            user=self.db_settings['user'], 
            password=self.db_settings['password'],
            port=self.db_settings['port']
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
        return results

    def get_query_string(self, query):
        with psycopg.connect(
            dbname=self.db_settings['dbname'],
            host=self.db_settings['host'],
            user=self.db_settings['user'], 
            password=self.db_settings['password'],
            port=self.db_settings['port']
        ) as conn:
            query_str = query.as_string(conn)
        return query_str