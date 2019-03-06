#!/usr/bin/env python
import configparser
import json
import os
from urllib.parse import urlparse
import pika
from pymongo import MongoClient
from trainer.train import train

# ---------------------------------------------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
epochs = cfg.getint('training', 'TRAINING_EPOCHS')
validation_split = cfg.getfloat('training', 'VALIDATION_SPLIT')
batch_size = cfg.getint('training', 'BATCH_SIZE')
save_graph = cfg.getboolean('training', 'SAVE_GRAPH')
use_logging = cfg.getboolean('training', 'USE_LOGGING')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1@ds143070.mlab.com:43070/cubeauth')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')
# --------------------------------------------------------------

# connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URI))
# channel = connection.channel()

# channel.queue_declare(queue='trainings')


def callback(ch, method, properties, body):
    # Training of the model is launched
    body = b'{"email": "julio.garciaperez@telefonica.com"}'

    pad, delta_t_ref = train(email=json.loads(body.decode('utf-8'))['email'], ep=epochs, val_split=validation_split, batch_size=batch_size, graph=False, logging=use_logging)

    # Connection to MongoDB is established
    client = MongoClient(MONGO_URI)
    # Getting a Database and parsing the name of the database from the MONGO_URI
    o = urlparse(MONGO_URI).path[1::]
    db = client[o]


    # Once training is finished, the user status that triggered training
    # is set to authenticable
    db.users.update_one(json.loads(body), {'$set': {'authenticable': True, 'required_movements': pad, 'time_ref': delta_t_ref}})

    #print(" [x] Train triggered by %r" % json.loads(body)['email'])

callback(1,1,1,1)
# channel.basic_consume(callback,
#                       queue='trainings',
#                       no_ack=True)

# print(' [*] Waiting for messages. To exit press CTRL+C')
# channel.start_consuming()


