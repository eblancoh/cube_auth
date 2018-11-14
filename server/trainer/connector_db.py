# usr/bin/env python

import os
import sqlite3

import pandas as pd

basedir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))
database_dir = os.path.join(basedir, 'database_dir')


def database_connection(database, table_name):
    """

    :return:
    """
    # Conectar con la base de datos
    # Using the connect function, which returns a Connection object:
    connection = sqlite3.connect(os.path.join(database_dir, database))
    # Pasar a pandas.DataFrame el conjunto de movimientos almacenados en la base de datos
    # Build the query to retrieve information from Movements database
    query = "select * from " + table_name + ";"# Movements;"
    try:
        moves = pd.read_sql_query(sql=query, con=connection)
        moves.stamp = pd.to_datetime(moves.stamp, format='%Y-%m-%dT%H:%M:%S.%fZ')
        # New query for users
        # query = "select * from Users;"
        # users = pd.read_sql_query(sql=query, con=connection)
        # Close the connection with database
        connection.close()
        # Get only those movements that belong to sequences started from random states of the cube
        moves = moves.loc[(moves.is_random == 1)]  # & moves.user.isin(valid_users)]

        # Replace NaNs in positioning with zeros
        moves = moves.fillna(0)
        return moves
    except BaseException:
        print('An error occurred! Please, check if queried table exists or is correctly built')

