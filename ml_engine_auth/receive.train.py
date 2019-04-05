#!/usr/bin/env python
import configparser
import json
import os
from urllib.parse import urlparse
import pika
from pymongo import MongoClient
from engine import training_dataframe, obtain_features, user_to_binary, ml_engine_training, save_scaling, load_scaling

# ---------------------------------------------------------------
# Some Configuration 
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
RABBIT_URI = os.environ.get('RABBIT_URI', 'localhost')
basedir = os.path.abspath(os.path.dirname(__file__))
checkpoint_path = os.path.join(basedir, 'checkpoints')
split = 1
model = 'svc'
# model = 'logRegr'
# --------------------------------------------------------------

connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URI))
channel = connection.channel()
channel.queue_declare(queue='trainings')

def callback(ch, method, properties, body):
    # Training of the model is launched
    df = training_dataframe(mongodb_uri=MONGO_URI, split=split)
    users = df['user_email'].unique()

    # All the checkpoints to be stored in checkpoints path
    os.chdir(checkpoint_path)
    for user in users:
        data = user_to_binary(df, user)
        # Aplicamos estandarización. Se guardará un fichero de estandarización en la carpeta checkpoints
        # data = save_scaling(data)
        X_train, X_test, Y_train, Y_test = obtain_features(dataframe=data)

        X_train = save_scaling(X_train)
        # Normalizamos el test dataset de acuerdo al training dataset 
        X_test = load_scaling(X_test)

        ml_engine_training(X_train, X_test, Y_train, Y_test, user, model=model)
        
        print('Training for user ', user, ' finished!')
    os.chdir(basedir)

    # Connection to MongoDB is established
    client = MongoClient(MONGO_URI),
    # Getting a Database and parsing the name of the database from the MONGO_URI
    o = urlparse(MONGO_URI).path[1::]
    db = client[o]
    # Once training is finished, the user status that triggered training
    # is set to authenticable
    db.users.update_one(json.loads(body), {'$set': {'authenticable': True}})
    #print(" [x] Train triggered by %r" % json.loads(body)['email'])

channel.basic_consume(callback,
                      queue='trainings',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()


