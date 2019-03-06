# usr/bin/env python

import configparser
import os
import ast
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from .connector_db2 import database_connection

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))

# Cargando del fichero de configuración los parámetros necesarios
config = configparser.ConfigParser()
config.read(os.path.join(current_dir, '..', '..', 'config', 'cfg.ini'))
pad = config.getint('training', 'NUM_MOVES')
no_solvers = config.getint('deep_model', 'NO_SOLVERS')


def input_to_train_lstm(database='CubeAuth.sqlite', table_name='Movements2'):
    """

    :return:
    """

    stamps = list()
    turns = list()
    YawPitchRoll = list()
    solvers = list()

    # try:
    # connection to database
    moves = database_connection(database, table_name)

    # agrupamos por usuario y por número de solución al que pertenecen los movimientos
    grouped = moves.groupby(['user_id', 'solution'])
    for name, group in grouped:
        stamp = group['stamp'].values
        seq = group['turn'].values
        angles = group['YawPitchRoll'].values
        user = name[0]

        # Transformamos los timestamps a segundos desde el inicio de cada secuencia
        stamp = (stamp - stamp[0]) / np.timedelta64(1, 's')

        # Transformamos la secuencia de giros en base al bit que nos interesa
        seq = np.array([int(k[-8:-6], 16) for k in seq])

        # Procesamos los ángulos obtenidos. Están almacenados en base de datos como string
        pos_byte_list = list()
        for k in angles:
            if k is None:
                pos_byte_list.append([0] * moves.frequency.unique()[0])
            else:
                pos_byte_list.append(ast.literal_eval(k))
        pos_byte_list = np.array(pos_byte_list)
        # Esto debe ser un tensor con dimensiones (num_moves, moves.frequency.unique()[0], 3)
        # Es decir, (número de movimientos de una secuencia, número de posicionamientos por movimimento, yaw+pitch+roll)
        zxy = list() 
        for i in pos_byte_list:
            ypr = list()
            for j in i:
                if j == '0':
                    yaw = 0
                    pitch = 0
                    roll = 0
                else: 
                    yaw = int(j[0:4], 16) / 100
                    pitch = int(j[4:8], 16) / 100
                    roll = int(j[8:12], 16) / 100
                ypr.append(np.array([yaw, pitch, roll]))
            zxy.append(np.array(ypr))
        zxy = np.array(zxy)

        # Si el número de movimientos en el que se ha resuelto el cubo es menor que pad, 
        # debemos rellenar hasta pad
        if pad > zxy.shape[0]:
            aux_zeros = np.zeros((pad - zxy.shape[0], zxy.shape[1], zxy.shape[2])) 
            zxy = np.concatenate((zxy, aux_zeros))
        else:
            # Por el contrario, haremos un cropping del tensor de posicionamiento en el caso de que tengamos un 
            zxy = zxy[0:pad]

        # Almacenamos los valores en las listas definidas
        stamps.append(stamp)
        turns.append(seq)
        YawPitchRoll.append(zxy)
        solvers.append(user)

    # Truncate and pad input sequences. Force formats also.
    # Reshape is required if certain architecture is chosen for data processing
    # reshape to elem.shape to (?, elem.shape, 1)
    # use truncating='pre' to pad from the end of the sequence
    # use truncating='post' to pad from the begin of the sequence
    stamps = np.expand_dims(sequence.pad_sequences(stamps,
                                                    maxlen=pad,
                                                    truncating='post',
                                                    dtype="float32"), axis=2)
    turns = np.expand_dims(sequence.pad_sequences(turns,
                                                    maxlen=pad,
                                                    truncating='post',
                                                    dtype="float32"), axis=2)
    YawPitchRoll = np.array(YawPitchRoll)
    
    # Generamos las labels de cada secuencia
    labels = to_categorical(solvers, num_classes=no_solvers)

    return [stamps, turns, YawPitchRoll], labels

    # except BaseException:
        # print('Something went wrong! Check connection to database')


# Get only those users who have more than auth_threshold solutions to train the model
# valid_users = users[users.solutions > users.auth_threshold].user.unique()
# Filtrar por tipo de cubo
# Diferentes entrenamientos en función de ello?


def input_to_test_lstm(database='CubeAuth.sqlite', table_name='TestMovements2'):
    """

    :return:
    """

    #try:
    # connection to database
    moves = database_connection(database, table_name)

    # agrupamos por usuario y por númmero de solución al que pertenecen los movimientos
    # grouped = moves.groupby(['user_id', 'solution'])
    # for name, group in grouped:
    stamp = moves['stamp'].values
    seq = moves['turn'].values
    YawPitchRoll = moves['YawPitchRoll'].values

    pos_byte_list = list()
    for k in YawPitchRoll:
        if k is None:
            pos_byte_list.append([0] * moves.frequency.unique()[0])
        else:
            pos_byte_list.append(ast.literal_eval(k))

    pos_byte_list = np.array(pos_byte_list)

    zxy = list()
    for i in pos_byte_list:
        ypr = list()
        for j in i:
            if j == '0':
                yaw = 0
                pitch = 0
                roll = 0
            else:
                yaw = int(j[0:4], 16) / 100
                pitch = int(j[4:8], 16) / 100
                roll = int(j[8:12], 16) / 100
            ypr.append(np.array([yaw, pitch, roll]))
        zxy.append(np.array(ypr))
    zxy = np.array(zxy)

    # Transformamos los timestamps a segundos desde el inicio de cada secuencia
    stamp = (stamp - stamp[0]) / np.timedelta64(1, 's')

    # Transformamos la secuencia de giros en base al bit que nos interesa
    seq = np.array([int(k[-8:-6], 16) for k in seq])


    # Reshape is required if certain architecture is chosen for data processing
    # reshape to elem.shape to (1, elem.shape, 1) for those processed by LSTM
    # and reshape to (1, pad, moves.frequency.unique()[0], 3) those to be processed by
    # convolutioal branch

    stamp = np.reshape(a=stamp, newshape=(1, pad, 1))
    seq = np.reshape(a=seq, newshape=(1, pad, 1))
    zxy = np.reshape(a=zxy, newshape=(1, pad, moves.frequency.unique()[0], 3))

    # Generamos las labels de cada secuencia
    # label = to_categorical(user, num_classes=no_solvers)

    return [stamp, seq, zxy]#, label

    # except BaseException:
        # print('Something went wrong! Check connection to database')
