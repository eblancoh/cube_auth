#!/usr/bin/env python
import pika
import configparser
import os
import json
from trainer import model_builder2
from trainer.mongodb import feed_training_data, feed_testing_data
from pymongo import MongoClient
from urllib.parse import urlparse
from keras import backend as K
import time
from keras.models import load_model

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
prob_threshold = cfg.getfloat('predict', 'PROB_THRESHOLD')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')

#--------------------------------------------------------------#

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='testing queue')

def callback(ch, method, properties, body):
    # Testeo

    auth = {"success": False}

    # Obtenemos las secuencias a testear
    testing_inputs = feed_testing_data(body=json.loads(body.decode('utf-8'))['movements'])
    login_id = json.loads(body.decode('utf-8'))['id']

    auth["id"] = login_id

    # Necesitamos el mapeo etiqueta => integer en un dictionary
    mapper = feed_training_data()[2]
    q = mapper[json.loads(body.decode('utf-8'))['email']]

    start_time = time.time()
    
    # Cargamos el modelo
    model = load_model(os.path.join(basedir, 'checkpoints', 'weights.best.h5'))

    print("--- Model loading took %s seconds ---" % (time.time() - start_time))
    
    # The label after the sequence is predicted
    probability, y_pred = model_builder2.model_predict(model=model, inputs=[testing_inputs[0], testing_inputs[1], testing_inputs[2]], verbose=0)

    auth["predict"] = []
    prob_y_pred = probability[0][y_pred]
    ret = {
                "label": int(y_pred),
                "probability": float(prob_y_pred)
            }
    auth["predict"].append(ret)

    if y_pred == q and prob_y_pred > prob_threshold:
                # If the predicted label is the same as the provided with the json file:
                auth["success"] = True

    # Devolución del resultado
    print(auth)


channel.basic_consume(callback,
                      queue='testing queue',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

