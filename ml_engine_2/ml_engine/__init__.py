#!/usr/bin/env python
"""
import configparser
import json
import os
from urllib.parse import urlparse
import pika
from pymongo import MongoClient
from trainer.train import train

# ---------------------------------------------------------------
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')

if __name__ == "__main__":
    train(email='julio.garciaperez@telefonica.com', ep=10, val_split=0.1, batch_size=1, graph=True, logging=True)
"""

#!/usr/bin/env python
import configparser
import json
import os
import time

import pika
from keras import backend as K
from keras.models import load_model
from trainer import datasetbuilder
from trainer import modelbuilder
from pymongo import MongoClient
from urllib.parse import urlparse

# ---------------------------------------------------------------
# Base directory path is defined
basedir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
prob_threshold = cfg.getfloat('predict', 'PROB_THRESHOLD')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')
# --------------------------------------------------------------

auth = {"success": False}

# Primero necesito los factores de escalabilidad de la red 
# pad y la mediana de dt para el usuario 

body=b'{"email":"julio.garciaperez@telefonica.com","movements":[{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:45.888Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:46.675Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:47.260Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:47.924Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:49.172Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:49.758Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"63","is_random":true,"timestamp":"2019-01-15T10:48:50.602Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:51.254Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:51.974Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"43","is_random":true,"timestamp":"2019-01-15T10:48:52.649Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"11","is_random":true,"timestamp":"2019-01-15T10:48:53.381Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:53.921Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"43","is_random":true,"timestamp":"2019-01-15T10:48:54.832Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:55.326Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:56.238Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:57.116Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"63","is_random":true,"timestamp":"2019-01-15T10:48:57.881Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:58.533Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:59.332Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:59.984Z","yaw_pitch_roll":[]}],"id":"5c3dba9da95921001691a857"}'

#pad, delta_t_median = datasetbuilder.feed_training_data(email=json.loads(body.decode('utf-8'))['email'])[-2:]
# Sequence to be tested is preprocesssed to feed neural network
testing_inputs = datasetbuilder.feed_testing_data(body=json.loads(body.decode('utf-8'))['movements'], email=json.loads(body.decode('utf-8'))['email'], pad=pad, delta_t_median=delta_t_median)
# Login_id is extracted
login_id = json.loads(body.decode('utf-8'))['id']

auth["id"] = login_id

start_time = time.time()
print('Model loading from existing checkpoint in progress')

# The model is loaded from checkpoint
K.clear_session()
model = load_model(os.path.join(basedir, 'checkpoints', 'weights.best.' + json.loads(body.decode('utf-8'))['email'] + '.h5'))

print("Model loading finished. Process took %s seconds ---" % (time.time() - start_time))

# The label after the sequence is predicted
# probability, 
y_pred = modelbuilder.model_predict(model=model,
                                    inputs=[testing_inputs[0], testing_inputs[1], testing_inputs[2]],
                                    verbose=0)[0]
auth["score"] = y_pred
# The answer to be provided after testing is built

if y_pred * 100 > prob_threshold:
    auth["success"] = True

# auth result is returned
print(auth)
