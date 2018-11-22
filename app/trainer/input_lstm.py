# usr/bin/env python

import configparser
import os

import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from trainer.connector_db import database_connection

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))

# Cargando del fichero de configuración los parámetros necesarios
config = configparser.ConfigParser()
config.read(os.path.join(current_dir, '..', '..', 'config', 'cfg.ini'))
pad = config.getint('training', 'NUM_MOVES')
no_solvers = config.getint('deep_model', 'NO_SOLVERS')


def input_to_train_lstm(database='CubeAuth.sqlite', table_name='Movements'):
    """

    :return:
    """

    stamps = list()
    shifts = list()
    angles_x = list()
    angles_y = list()
    angles_z = list()
    solvers = list()

    try:
        # connection to database
        moves = database_connection(database, table_name)

        # agrupamos por usuario y por númmero de solución al que pertenecen los movimientos
        grouped = moves.groupby(['user_id', 'solution'])
        for name, group in grouped:
            stamp = group['stamp'].values
            shift = group['data_code']
            data_x = group['data_x'].values
            data_y = group['data_y'].values
            data_z = group['data_z'].values
            user = name[0]

            # Possible values considered for shift information provided
            shift_map = {'0xA0': 1, '0xA1': 2, '0xA2': 3, '0xA3': 4,
                         '0xB0': 5, '0xB1': 6, '0xB2': 7, '0xB3': 8,
                         '0xC0': 9, '0xC1': 10, '0xC2': 11, '0xC3': 12,
                         '0xD0': 13, '0xD1': 14, '0xD2': 15, '0xD3': 16,
                         '0xE0': 17, '0xE1': 18, '0xE2': 19, '0xE3': 20,
                         '0xF0': 21, '0xF1': 22, '0xF2': 23, '0xF3': 24}

            # Map the values of shifts again the defined dictionary
            shift = shift.map(shift_map).values
            # Transformamos los timestamps a segundos desde el inicio de cada secuencia
            stamp = (stamp - stamp[0]) / np.timedelta64(1, 's')

            # Almacenamos los valores en las listas definidas
            stamps.append(stamp)
            shifts.append(shift)
            angles_x.append(data_x)
            angles_y.append(data_y)
            angles_z.append(data_z)
            solvers.append(user)

        # Truncate and pad input sequences. Force formats also.
        # Reshape is required if certain architecture is chosen for data processing
        # reshape to elem.shape to (?, elem.shape, 1)
        # use truncating='pre' to pad from the end of the sequence
        # use truncating='post' to pad from the begin of the sequence
        stamps = np.expand_dims(sequence.pad_sequences(stamps,
                                                       maxlen=pad,
                                                       truncating='post',
                                                       dtype="float32"),
                                axis=2)
        shifts = np.expand_dims(sequence.pad_sequences(shifts,
                                                       maxlen=pad,
                                                       truncating='post',
                                                       dtype="float32"),
                                axis=2)
        angles_x = np.expand_dims(sequence.pad_sequences(angles_x,
                                                         maxlen=pad,
                                                         truncating='post',
                                                         dtype="float32"),
                                  axis=2)
        angles_y = np.expand_dims(sequence.pad_sequences(angles_y,
                                                         maxlen=pad,
                                                         truncating='post',
                                                         dtype="float32"),
                                  axis=2)
        angles_z = np.expand_dims(sequence.pad_sequences(angles_z,
                                                         maxlen=pad,
                                                         truncating='post',
                                                         dtype="float32"),
                                  axis=2)

        # Generamos las labels de cada secuencia
        labels = to_categorical(solvers, num_classes=no_solvers)

        return [stamps, shifts, angles_x, angles_y, angles_z], labels

    except BaseException:
        print('Something went wrong! Check connection to database')


# Get only those users who have more than auth_threshold solutions to train the model
# valid_users = users[users.solutions > users.auth_threshold].user.unique()
# Filtrar por tipo de cubo
# Diferentes entrenamientos en función de ello?


def input_to_test_lstm(database='CubeAuth.sqlite', table_name='Test_Movements'):
    """

    :return:
    """

    try:
        # connection to database
        moves = database_connection(database, table_name)

        # agrupamos por usuario y por númmero de solución al que pertenecen los movimientos
        # grouped = moves.groupby(['user_id', 'solution'])
        # for name, group in grouped:
        stamp = moves['stamp'].values
        shift = moves['data_code']
        data_x = moves['data_x'].values
        data_y = moves['data_y'].values
        data_z = moves['data_z'].values
        # user = moves['user_id'].unique()

        # Possible values considered for shift information provided
        shift_map = {'0xA0': 1, '0xA1': 2, '0xA2': 3, '0xA3': 4,
                     '0xB0': 5, '0xB1': 6, '0xB2': 7, '0xB3': 8,
                     '0xC0': 9, '0xC1': 10, '0xC2': 11, '0xC3': 12,
                     '0xD0': 13, '0xD1': 14, '0xD2': 15, '0xD3': 16,
                     '0xE0': 17, '0xE1': 18, '0xE2': 19, '0xE3': 20,
                     '0xF0': 21, '0xF1': 22, '0xF2': 23, '0xF3': 24}

        # Map the values of shifts again the defined dictionary
        shift = shift.map(shift_map).values
        # Transformamos los timestamps a segundos desde el inicio de cada secuencia
        stamp = (stamp - stamp[0]) / np.timedelta64(1, 's')


        # Reshape is required if certain architecture is chosen for data processing
        # reshape to elem.shape to (1, elem.shape, 1)

        stamp = np.reshape(a=stamp, newshape=(1, pad, 1))
        shift = np.reshape(a=stamp, newshape=(1, pad, 1))
        data_x = np.reshape(a=stamp, newshape=(1, pad, 1))
        data_y = np.reshape(a=stamp, newshape=(1, pad, 1))
        data_z = np.reshape(a=stamp, newshape=(1, pad, 1))

        # Generamos las labels de cada secuencia
        # label = to_categorical(user, num_classes=no_solvers)

        return [stamp, shift, data_x, data_y, data_z]#, label

    except BaseException:
        print('Something went wrong! Check connection to database')

