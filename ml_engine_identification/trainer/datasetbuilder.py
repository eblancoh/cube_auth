# usr/bin/env python

import configparser
import itertools
import os
from urllib.parse import urlparse

import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from pymongo import MongoClient

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))
# Required parameters for data processing are uploaded
config = configparser.ConfigParser()
config.read(os.path.join(current_dir, '..', '..', 'config', 'cfg.ini'))
pad = config.getint('training', 'NUM_MOVES')
frequency = config.getint('training', 'POS_FREQ')
no_solvers = config.getint('deep_model', 'NO_SOLVERS')
angle_max = config.getint('database', 'ANGLE_MAX')
angle_min = config.getint('database', 'ANGLE_MIN')
turn_max = config.getint('database', 'TURN_MAX')
delta_t_max = config.getint('database', 'DELTA_TIME_MAX')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')


def feed_training_data():
    """
    Function to generate the training and validation dataset for training our model
    :return: [times, turns, phi_theta_psi], labels, mapper
    """
    # Making a Connection with MongoClient
    client = MongoClient(MONGO_URI)

    o = urlparse(MONGO_URI).path[1::]

    # Getting a Database and parsing the name of the database from the MongoURI
    db = client[o]

    # Querying in MongoDB and obtaining the result as variable
    movements_collection = list(db.movements.find())

    # Group by user and solving sequence
    keys = list()
    groups = list()
    for key, group in itertools.groupby(movements_collection, key=lambda x: (x['email'], x['sequence'])):
        keys.append({'email': key[0], 'sequence': key[1]})
        groups.append(list(group))

    # Information is obtained from movements_collection
    times = list()
    turns = list()
    X = list()
    Y = list()
    Z = list()
    for k in groups:
        time = list()
        turn = list()
        x = list()
        y = list()
        z = list()
        for i in k:
            position = i['yaw_pitch_roll']
            time.append(i['timestamp'])
            turn.append(i['turn'])

            if position is not None and i['cube_type'] == '11paths':
                for j in position:
                    x.append(j['x'])
                    y.append(j['y'])
                    z.append(j['z'])
            else:
                for j in range(frequency):
                    x.append(0)
                    y.append(0)
                    z.append(0)

        dt = np.array(time).astype('datetime64')
        # Delta time between movements in milliseconds
        dt = np.hstack((0, np.diff(dt).astype('float32')))
        times.append(dt)

        turns.append(np.array(turn))

        # If the length of every sequence is not the expected, it should be completed with
        # zeros or shortened to the pad length
        if pad * frequency > len(x):
            x.extend([0] * (pad * frequency - len(x)))
            y.extend([0] * (pad * frequency - len(y)))
            z.extend([0] * (pad * frequency - len(z)))
        else:
            x = x[0:pad * frequency]
            y = y[0:pad * frequency]
            z = z[0:pad * frequency]

        X.append(np.array(x).reshape((pad, frequency)))
        Y.append(np.array(y).reshape((pad, frequency)))
        Z.append(np.array(z).reshape((pad, frequency)))

    # Truncate and pad input sequences. Force formats also.
    # Reshape is required if certain architecture is chosen for data processing
    # reshape to elem.shape to (?, elem.shape, 1)

    times = np.expand_dims(sequence.pad_sequences(times, maxlen=pad, dtype="float32"), axis=2)
    turns = np.expand_dims(sequence.pad_sequences(turns, maxlen=pad, dtype="float32"), axis=2)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    # Positioning tensor must have a shape (?, pad, frequency, channels = 3)
    phi_theta_psi = np.stack((X, Y, Z), axis=3)

    # All inputs are normalized between 0 and 1
    times = times / delta_t_max
    turns = turns / turn_max
    phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

    users = list()
    for k in keys:
        users.append(k['email'])

    # A dictionary is created so the users are mapped to an integer and the order of
    # that mapping is never lost
    uniques = np.unique(users, return_index=True)
    mapper = dict()
    for item in np.sort(uniques[1]):
        mapper.update({users[item]: len(mapper)})

    users_int = list()
    for item in users:
        users_int.append(mapper[item])

    # Labels for each sequence is obtained
    labels = to_categorical(users_int, num_classes=no_solvers)

    return [times, turns, phi_theta_psi], labels, mapper


def feed_testing_data(body):
    """
    Function to generate the testing sequence for testing our model
    :param body: array of movements in json format
    :return: [time, turn, phi_theta_psi]
    """
    time = list()
    turn = list()
    x = list()
    y = list()
    z = list()
    for i in body:
        time.append(i['timestamp'])
        turn.append(i['turn'])
        position = i['yaw_pitch_roll']

        if position is not None and i['cube_type'] == '11paths':
            for j in position:
                x.append(j['x'])
                y.append(j['y'])
                z.append(j['z'])
        else:
            for j in range(frequency):
                x.append(0)
                y.append(0)
                z.append(0)

    time = np.array(time).astype('datetime64')
    # Delta time between movements in milliseconds
    time = np.hstack((0, np.diff(time).astype('float32')))

    turn = np.array(turn).astype('float32')

    x = np.array(x).reshape((pad, frequency))
    y = np.array(y).reshape((pad, frequency))
    z = np.array(z).reshape((pad, frequency))

    # Positioning tensor shall have a shape (1, pad, frequency, channels = 3)
    phi_theta_psi = np.stack((x, y, z), axis=2)

    # All inputs are normalized between 0 and 1
    time = time / delta_t_max
    turn = turn / turn_max
    phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

    time = np.reshape(a=time, newshape=(1, pad, 1))
    turn = np.reshape(a=turn, newshape=(1, pad, 1))
    phi_theta_psi = np.reshape(a=phi_theta_psi, newshape=(1, pad, frequency, 3))

    return [time, turn, phi_theta_psi]
