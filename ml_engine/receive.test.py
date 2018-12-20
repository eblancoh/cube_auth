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

# ---------------------------------------------------------------
# Base directory path is defined
basedir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
prob_threshold = cfg.getfloat('predict', 'PROB_THRESHOLD')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/cubeauthdb')
# --------------------------------------------------------------

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='testing queue')


def callback(ch, method, properties, body):

    # Testing starts

    auth = {"success": False}

    # Sequence to be tested is preprocesssed to feed neural network
    testing_inputs = datasetbuilder.feed_testing_data(body=json.loads(body.decode('utf-8'))['movements'])
    # Login_id is extracted
    login_id = json.loads(body.decode('utf-8'))['id']

    auth["id"] = login_id

    # The mapper from label to integer is required for testing
    mapper = datasetbuilder.feed_training_data()[2]
    # This will be the reference inteher to match during testing
    q = mapper[json.loads(body.decode('utf-8'))['email']]

    start_time = time.time()
    print('Model loading from existing checkpoint in progress')

    # The model is loaded from checkpoint
    K.clear_session()
    model = load_model(os.path.join(basedir, 'checkpoints', 'weights.best.h5'))

    print("Model loading finished. Process took %s seconds ---" % (time.time() - start_time))

    # The label after the sequence is predicted
    probability, y_pred = modelbuilder.model_predict(model=model,
                                                     inputs=[testing_inputs[0], testing_inputs[1], testing_inputs[2]],
                                                     verbose=0)
    # The answer to be provided after testing is built
    auth["predict"] = []
    prob_y_pred = probability[0][y_pred]
    ret = {
        "label": int(y_pred),
        "probability": round(float(prob_y_pred), 3)
    }
    auth["predict"].append(ret)

    if y_pred == q and prob_y_pred > prob_threshold:
        # If the predicted label is the same as the provided with the json file
        # and the probability threshold is bigger than defined:
        auth["success"] = True

    # auth result is returned
    print(auth)


channel.basic_consume(callback,
                      queue='testing queue',
                      no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
