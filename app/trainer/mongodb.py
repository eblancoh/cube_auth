import pymongo
from pymongo import MongoClient
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from itertools import groupby
from operator import itemgetter
import itertools
import configparser
import os
from urllib.parse import urlparse


# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))
# Cargando del fichero de configuración los parámetros necesarios
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
  # Making a Connection with MongoClient
  client = MongoClient(MONGO_URI)

  o=urlparse(MONGO_URI).path[1::]
  # Getting a Database and parsing the name of the database from the MongoURI
  db = client[o]

  # Querying in MongoDB and obtaining the result as variable
  user_collection = list(db.users.find())
  movements_collection = list(db.movements.find())

  # To delete collections:
  # db.users.delete_many(filter={})
  # db.movements.delete_many(filter={})

  # Agrupar por usuario y por secuencia de resolución
  keys = list()
  groups = list()
  for key, group in itertools.groupby(movements_collection, key=lambda x:(x['email'], x['sequence'])):
    keys.append({'email': key[0], 'sequence': key[1]})
    groups.append(list(group))

  # Empezamos a extraer la información de la base de datos de MongoDB
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

      for j in position:
        x.append(j['x'])
        y.append(j['y'])
        z.append(j['z'])

    dt = np.array(time).astype('datetime64')
    # Transformamos los timestamps a segundos desde el inicio de cada secuencia
    dt = np.hstack((0, np.diff(dt).astype('float32')))
    times.append(dt)

    turns.append(np.array(turn))

    # Si nos encontramos con que la longitud de cada secuencia no es la que debería, 
    # bien la completamos o la acortamos a la dimensión de padding elegida
    if pad * frequency > len(x):
      x.extend([0]*(pad * frequency - len(x)))
      y.extend([0]*(pad * frequency - len(y)))
      z.extend([0]*(pad * frequency - len(z)))
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
  X = np.array(X); Y = np.array(Y); Z = np.array(Z)
  # El tensor de posicionamiento debe tener una dimensión (?, pad, frequency, canales)
  phi_theta_psi=np.stack((X, Y, Z), axis=3)

  # Normalicemos los inputs entre 0 y 1:
  times = times / delta_t_max 
  turns = turns / turn_max
  phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

  users = list()
  for k in keys:
    users.append(k['email'])

  # Creamos un diccionario sin que se pierda el orden para los usuarios
  uniques = np.unique(users, return_index=True)
  mapper = dict()
  for item in np.sort(uniques[1]):
    mapper.update({users[item]: len(mapper)})

  users_int = list()
  for item in users:
    users_int.append(mapper[item])

  # Generamos las labels de cada secuencia
  labels = to_categorical(users_int, num_classes=no_solvers)

  return [times, turns, phi_theta_psi], labels, mapper


def feed_testing_data(body):
  
  time = list()
  turn = list()
  x = list()
  y = list()
  z = list()
  for i in body:
    time.append(i['timestamp'])
    turn.append(i['turn'])
    position = i['yaw_pitch_roll']

    for j in position:
        x.append(j['x'])
        y.append(j['y'])
        z.append(j['z'])

  time = np.array(time).astype('datetime64')
  # Transformamos los timestamps a segundos desde el inicio de cada secuencia
  time = np.hstack((0, np.diff(time).astype('float32')))

  turn = np.array(turn).astype('float32')
  
  x = np.array(x).reshape((pad, frequency))
  y = np.array(y).reshape((pad, frequency))
  z = np.array(z).reshape((pad, frequency))

  # El tensor de posicionamiento debe tener una dimensión (?, pad, frequency, canales)
  phi_theta_psi=np.stack((x, y, z), axis=2)

  # Normalicemos los inputs entre 0 y 1:
  time = time / delta_t_max 
  turn = turn / turn_max
  phi_theta_psi = (phi_theta_psi - angle_min) / (angle_max - angle_min)

  time = np.reshape(a=time, newshape=(1, pad, 1))
  turn = np.reshape(a=turn, newshape=(1, pad, 1))
  phi_theta_psi = np.reshape(a=phi_theta_psi, newshape=(1, pad, frequency, 3))

  return [time, turn, phi_theta_psi]
