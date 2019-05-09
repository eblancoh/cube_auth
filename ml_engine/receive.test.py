#!/usr/bin/env python
import os
from urllib.parse import urlparse
from pymongo import MongoClient
from bson.objectid import ObjectId
import pika
import numpy as np
from engine import testing_dataframe, model_testing, load_scaling

# ---------------------------------------------------------------
# Base directory path is defined
# Some Configuration
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')
basedir = os.path.abspath(os.path.dirname(__file__))
checkpoint_path = os.path.join(basedir, 'checkpoints')
prob_threshold = 0.5

# Following models are supported
models = ['logRegr', 'svc', 'RandomForest']
# --------------------------------------------------------------

connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URI))
channel = connection.channel()

channel.queue_declare(queue='logins')

def callback(ch, method, properties, body):

    # Testing starts

    auth = {"success": False}

    user, df, login_id = testing_dataframe(body)

    # print(body)
    # print(df)
    
    auth["id"] = login_id

    # Vamos a comprobar la probabilidad que nos daría cada uno de los modelos soportados 
    # en el presente motor. Iteramos sobre la lista de los modelos de arriba
    probs = list()
    for model in models:
        os.chdir(checkpoint_path)
        # Se lanza el testeo de la probabilidad
        # Ya normalizamos el contenido procesado del body en la función model_testing().
        # salvo que el modelo sea random forest
        probs.append(model_testing(testeo=df, user=user, model=model))
        os.chdir(basedir)

    # Sólo nos interesamos por la probabilidad dada por models = 'RandomForest'
    probability = probs[2]
    # probability = np.median(probs)
    # probability = np.max(probs)
    # The answer to be provided after testing is built
    auth["predict"] = []
    ret = {
        "probability": round(float(probability), 3)
    }
    auth["predict"].append(ret)

    if probability >= prob_threshold:
        # If the predicted label is the same as the provided with the json file
        # and the probability threshold is bigger than defined:
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
    db.logins.update_one({ '_id': ObjectId(auth['id']) }, {'$set': {'success': auth['success'] , 'probability': probability}})


channel.basic_consume(callback,
                      queue='logins',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
