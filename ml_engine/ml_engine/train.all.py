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
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
checkpoint_path = os.path.join(basedir, 'checkpoints')
# --------------------------------------------------------------

# Connection to MongoDB is established
client = MongoClient(MONGO_URI)
# Getting a Database and parsing the name of the database from the MONGO_URI
o = urlparse(MONGO_URI).path[1::]
db = client[o]

# Querying in MongoDB and obtaining the result as variable
users_collection = list(db.users.find())

for item in users_collection:
    if item['authenticable'] == True:
        # Si los usuarios son autenticables, entonces lanzamos los entrenamientos
        # Training of the model is launched
        # Si no existe el checkpoint, pero el usuario es autenticable, hay que entrenar y definir los parámetros que caracterizan el modelo
        # Lo lanzamos con los valores caracteríticos de cada usuario leyéndolo de la base de datos, por lo que first = False 
        train(email=item['email'], ep=epochs, val_split=validation_split, batch_size=batch_size, graph=False, logging=use_logging, first=False)
