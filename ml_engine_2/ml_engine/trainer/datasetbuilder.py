# usr/bin/env python

import configparser
import itertools
import os
from urllib.parse import urlparse
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from pymongo import MongoClient
import sys

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))
# Required parameters for data processing are uploaded
config = configparser.ConfigParser()
config.read(os.path.join(current_dir, '..', '..', 'config', 'cfg.ini'))
frequency = config.getint('training', 'POS_FREQ')
angle_max = config.getint('database', 'ANGLE_MAX')
angle_min = config.getint('database', 'ANGLE_MIN')
turn_max = config.getint('database', 'TURN_MAX')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1@ds143070.mlab.com:43070/cubeauth')


def feed_training_data(email, first):
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
    for key, group in itertools.groupby(movements_collection, key=lambda x: (x['user_email'], x['sequence'])):
        keys.append({'user_email': key[0], 'sequence': key[1]})
        groups.append(list(group))
    
    if first:
        # Para conseguir un tamaño de secuencia característico del usuario especificado por "email"
        keys_locator=list()
        for i in range(len(keys)):
            if keys[i]['user_email'] == email:
                keys_locator.append(i)

        pad_list = list()
        for i in keys_locator:
            pad_list.append(groups[i].__len__())              
        
        try: 
            pad = np.int(np.median(pad_list)) + np.int(0.1 * np.median(pad_list))
        except ValueError:
            sys.exit("No valid user has been found in database for movements query")

        # Para conseguir el máximo tiempo que un usuatamaño de secuencia característico del usuario especificado por "email"
        times_list = list()
        for i in keys_locator:
            aux_array = []
            for item in groups[i]:
                aux_array.append(item['timestamp'])
            #times_list.append(np.hstack((0, np.diff(np.array(aux_array).astype('datetime64')).astype('float32'))))
            times_list.append(np.diff(np.array(aux_array).astype('datetime64[ms]')).astype('float32'))
        
        # La referencia temporal el la mediana de la distribución más la desviación típica
        delta_t_ref = np.median(np.concatenate(times_list).ravel().tolist()) + np.std(np.concatenate(times_list).ravel().tolist())
    else:

        


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
            try: 
                # Elegimos un número 'frequency' de posicionamientos entre movimientos 
                # de manera aleatoria entre todos aquellos que se han recibido tras un giro 
                pos_seed = np.sort(np.random.choice(len(position), frequency, replace=False))

                if position is not None and i['cube_type'] == '11paths':
                    for j in np.array(position)[pos_seed]:
                        x.append(j['x'])
                        y.append(j['y'])
                        z.append(j['z'])
                else:
                    for j in range(frequency):
                        x.append(0)
                        y.append(0)
                        z.append(0)
            except ValueError:
                # Si se han recibido menos de los demandados, nos quedamos 
                # con todos y rellenamos con 0's
                if position is not None and i['cube_type'] == '11paths':
                    for j in range(frequency):
                        if j < len(position):
                            x.append(position[j]['x'])
                            y.append(position[j]['y'])
                            z.append(position[j]['z'])
                        else:
                            x.append(0)
                            y.append(0)
                            z.append(0) 
                else:
                    for j in range(frequency):
                        x.append(0)
                        y.append(0)
                        z.append(0)

        dt = np.array(time).astype('datetime64[ms]')
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
    times = times / delta_t_ref
    # If an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
    times = np.clip(a=times, a_min = 0, a_max = 1)
    turns = turns / turn_max
    phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

    users = list()
    for k in keys:
        users.append(k['user_email'])
    
    labels = np.where(np.array(users)==email, 1, 0)

    return [times, turns, phi_theta_psi], labels, pad, delta_t_ref


def feed_testing_data(body, email):
    """
    Function to generate the testing sequence for testing our model
    :param body: array of movements in json format
    :return: [time, turn, phi_theta_psi]
    """

    # Obtenemos de base de datos el tamaño del pad de la secuencia
    client = MongoClient(MONGO_URI)
    o = urlparse(MONGO_URI).path[1::]
    # Getting a Database and parsing the name of the database from the MongoURI
    db = client[o]
    # Querying in MongoDB and obtaining the result as variable
    users_collection = list(db.users.find())

    for key, group in itertools.groupby(users_collection, key=lambda x: (x['email'])):
        for item in list(group):
            if item['email'] == email:
                pad = item['required_movements']
                delta_t_ref = item['time_ref']
            
    time = list()
    turn = list()
    x = list()
    y = list()
    z = list()
    for i in body:
        time.append(i['timestamp'])
        turn.append(i['turn'])
        position = i['yaw_pitch_roll']

        try:
            # Elegimos un número 'frequency' de posicionamientos entre movimientos 
            # de manera aleatoria entre todos aquellos que se han recibido tras un giro 

            pos_seed = np.sort(np.random.choice(len(position), frequency, replace=False))

            if position is not None and i['cube_type'] == '11paths':
                for j in np.array(position)[pos_seed]:
                    x.append(j['x'])
                    y.append(j['y'])
                    z.append(j['z'])
            else:
                for j in range(frequency):
                    x.append(0)
                    y.append(0)
                    z.append(0)
        except ValueError:
            # Si se han recibido menos de los demandados, nos quedamos 
            # con todos y rellenamos con 0's
            if position is not None and i['cube_type'] == '11paths':
                for j in range(frequency):
                    if j < len(position):
                        x.append(position[j]['x'])
                        y.append(position[j]['y'])
                        z.append(position[j]['z'])
                    else:
                        x.append(0)
                        y.append(0)
                        z.append(0) 
            else:
                for j in range(frequency):
                    x.append(0)
                    y.append(0)
                    z.append(0)
    
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

    time = np.array(time).astype('datetime64[ms]')
    # Delta time between movements in milliseconds
    time = np.hstack((0, np.diff(time).astype('float32')))

    turn = np.array(turn).astype('float32')

    if len(turn) > pad:
        time = time[0:pad]
        turn = turn[0:pad]
    else:
        time = np.hstack((time, [0] * (pad - len(turn))))
        turn = np.hstack((turn, [0] * (pad - len(turn))))


    x = np.array(x).reshape((pad, frequency))
    y = np.array(y).reshape((pad, frequency))
    z = np.array(z).reshape((pad, frequency))

    # Positioning tensor shall have a shape (1, pad, frequency, channels = 3)
    phi_theta_psi = np.stack((x, y, z), axis=2)

    # All inputs are normalized between 0 and 1
    time = time / delta_t_ref
    # If an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
    time = np.clip(a=time, a_min = 0, a_max = 1)
    turn = turn / turn_max
    phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

    time = np.reshape(a=time, newshape=(1, pad, 1))
    turn = np.reshape(a=turn, newshape=(1, pad, 1))
    phi_theta_psi = np.reshape(a=phi_theta_psi, newshape=(1, pad, frequency, 3))

    return [time, turn, phi_theta_psi]

