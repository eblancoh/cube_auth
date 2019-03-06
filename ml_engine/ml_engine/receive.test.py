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
from trainer.modelbuilder import sensitivity, specificity, precision, recall
from pymongo import MongoClient
from urllib.parse import urlparse
from bson.objectid import ObjectId

# ---------------------------------------------------------------
# Base directory path is defined
basedir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
prob_threshold = cfg.getfloat('predict', 'PROB_THRESHOLD')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')
# --------------------------------------------------------------

connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URI))
channel = connection.channel()

channel.queue_declare(queue='logins')


def callback(ch, method, properties, body):

    # Testing starts

    auth = {"success": False}

    # body=b'{"email":"julio.garciaperez@telefonica.com","movements":[{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:45.888Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:46.675Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:47.260Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:47.924Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:49.172Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:49.758Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"63","is_random":true,"timestamp":"2019-01-15T10:48:50.602Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:51.254Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:51.974Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"43","is_random":true,"timestamp":"2019-01-15T10:48:52.649Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"11","is_random":true,"timestamp":"2019-01-15T10:48:53.381Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:53.921Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"43","is_random":true,"timestamp":"2019-01-15T10:48:54.832Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"61","is_random":true,"timestamp":"2019-01-15T10:48:55.326Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:56.238Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"31","is_random":true,"timestamp":"2019-01-15T10:48:57.116Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"63","is_random":true,"timestamp":"2019-01-15T10:48:57.881Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"51","is_random":true,"timestamp":"2019-01-15T10:48:58.533Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"13","is_random":true,"timestamp":"2019-01-15T10:48:59.332Z","yaw_pitch_roll":[]},{"email":"julio.garciaperez@telefonica.com","cube_type":"generic","turn":"23","is_random":true,"timestamp":"2019-01-15T10:48:59.984Z","yaw_pitch_roll":[]}],"id":"5c3dba9da95921001691a857"}'
    # Primero necesito los factores de escalabilidad de la red 
    # pad y la mediana de dt para el usuario 

    # pad, delta_t_ref = datasetbuilder.feed_training_data(email=json.loads(body.decode('utf-8'))['email'])[-2:]
    # delta_t_ref = datasetbuilder.feed_training_data(email=json.loads(body.decode('utf-8'))['email'])[-1]

    # Sequence to be tested is preprocesssed to feed neural network
    testing_inputs = datasetbuilder.feed_testing_data(body=json.loads(body.decode('utf-8'))['movements'], email=json.loads(body.decode('utf-8'))['email'])
    # Login_id is extracted
    login_id = json.loads(body.decode('utf-8'))['id']

    auth["id"] = login_id

    start_time = time.time()
    print('Model loading from existing checkpoint in progress')

    # The model is loaded from checkpoint
    K.clear_session()
    model = load_model(os.path.join(basedir, 'checkpoints', 'weights.best.' + json.loads(body.decode('utf-8'))['email'] + '.h5'), compile=True, custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall})

    print("Model loading finished. Process took %s seconds ---" % (time.time() - start_time))

    # The label after the sequence is predicted
    # probability, 
    y_pred = modelbuilder.model_predict(model=model,
                                        inputs=[testing_inputs[0], testing_inputs[1], testing_inputs[2]],
                                        verbose=0)[0]
    # The answer to be provided after testing is built

    auth["score"] = y_pred

    if y_pred * 100 > prob_threshold:
        auth["success"] = True

    # auth result is returned
    print(auth)

    # Connection to MongoDB is established
    client = MongoClient(MONGO_URI)
    # Getting a Database and parsing the name of the database from the MONGO_URI
    o = urlparse(MONGO_URI).path[1::]
    db = client[o]

    # Once training is finished, the user status that triggered training
    # is set to authenticable
    # db.logins.update_one({ '_id': ObjectId(auth['id']) }, {'$set': {'success': auth['success'] }})
    db.logins.update_one({ '_id': ObjectId(auth['id']) }, {'$set': {'success': auth['success'] , 'probability': float(round(y_pred, 2))}})

channel.basic_consume(callback,
                      queue='logins',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
