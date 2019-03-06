#!/usr/bin/env python
import pika
from trainer.train import train
import configparser
import os
import json
from pymongo import MongoClient
from urllib.parse import urlparse

#---------------------------------------------------------------#
basedir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
epochs = cfg.getint('training', 'TRAINING_EPOCHS')
validation_split = cfg.getfloat('training', 'VALIDATION_SPLIT')
batch_size = cfg.getint('training', 'BATCH_SIZE')
save_graph = cfg.getboolean('training', 'SAVE_GRAPH')
use_logging = cfg.getboolean('training', 'USE_LOGGING')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')

#--------------------------------------------------------------#

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='training queue')

def callback(ch, method, properties, body):
    # Se lanza entrenamiento del modelo
    train(ep=epochs, val_split=validation_split, batch_size=batch_size, graph=save_graph, logging=use_logging)
    
    # Conectarme a una base de datos
    client = MongoClient(MONGO_URI)
    # Getting a Database and parsing the name of the database from the MongoURI
    o=urlparse(MONGO_URI).path[1::]
    db = client[o]

    # Se actualiza el status del usuario que ha lanzado el entrenamiento a autenticable
    db.users.update_one(json.loads(body), {'$set': {'authenticable': True}})
    
    print(" [x] Train triggered by %r" % json.loads(body)['email'])

channel.basic_consume(callback,
                      queue='training queue',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

