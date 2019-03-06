#!/usr/bin/env python
import configparser
import json
import os
from urllib.parse import urlparse
import pika
from pymongo import MongoClient
import itertools
from trainer.datasetbuilder import feed_training_data
from trainer.modelbuilder import model_compiler, model_fitter, log_training, graph_model
from trainer.neuralnetwork_conv import rubik_deep_model
from pymongo import MongoClient
from urllib.parse import urlparse
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------
current_dir = os.path.abspath(os.path.dirname(__file__))
# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(current_dir, '..', 'config', 'cfg.ini'))
epochs = cfg.getint('training', 'TRAINING_EPOCHS')
validation_split = cfg.getfloat('training', 'VALIDATION_SPLIT')
batch_size = cfg.getint('training', 'BATCH_SIZE')
save_graph = cfg.getboolean('training', 'SAVE_GRAPH')
use_logging = cfg.getboolean('training', 'USE_LOGGING')
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
checkpoint_path = os.path.join(current_dir, 'checkpoints')
# --------------------------------------------------------------

# Connection to MongoDB is established
client = MongoClient(MONGO_URI)
# Getting a Database and parsing the name of the database from the MONGO_URI
o = urlparse(MONGO_URI).path[1::]
db = client[o]

# Querying in MongoDB and obtaining the result as variable
users_collection = list(db.users.find())

# fix random seed for reproducibility
seed = 314159
np.random.seed(seed)

# define n_splits-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

metrics_stats = []

for item in users_collection:
    tpr = []
    tnr = []
    ppv = []
    f1 = []
    val_dict = {'user': None, 'metrics': {'tpr': None,'tpr_std': None, 'tnr': None,'tnr_std': None, 'ppv': None, 'ppv_std': None, 'f1': None, 'f1_std': None}}
    if item['authenticable'] == True:
        pad = item['required_movements']
        delta_t_ref = item['time_ref']

        val_dict['user'] = item['email']
        
        features, labels= feed_training_data(item['email'], first=False)[0:2]

        # desired tradeoff between sensitivity and specificity
        t = 0.9

        # class weights can easily be incorporated into the loss by adding the 
        # following parameter to the fit function (assuming that 1 is the user doing the correct sequence):

        class_weight = {0: 1., 
                        1: t * (labels.__len__() - np.count_nonzero(labels))/np.count_nonzero(labels)}
        # "treat every instance of class 1 as N instances of class 0 times a correction factor" means 
        # that in your loss function you assign higher value to these instances. Hence, 
        # the loss becomes a weighted average, where the weight of each sample is specified
        #  by class_weight and its corresponding class.

        for train, test in kfold.split(features[0], labels):

                # Build the model according to the architecture specified
                model = rubik_deep_model(pad, yaw_pitch_roll=features[2])

                # The model shall be compiled before fitting process starts
                model_compiler(model=model, loss="binary_crossentropy",
                                optimizer="adam")
                                # , metrics=["accuracy"])

                print("Model compiled!\nCube Auth Model training in progress.")

                os.chdir(os.path.join(current_dir, "checkpoints"))
                # We fit the model to our data-set (i.e., our state and our labels )
                callback = model_fitter(model=model, inputs=[features[0][train], features[1][train], features[2][train]], labels=labels[train], email=item['email'], epochs=epochs,
                                        validation_split=validation_split, batch_size=batch_size, class_weight=class_weight, verbose=1)
                os.chdir(current_dir)

                scores = model.evaluate([features[0][test], features[1][test], features[2][test]], labels[test], verbose=0)


                tpr.append(scores[2])
                tnr.append(scores[3])
                ppv.append(scores[4])
                f1.append(2 * (ppv * tpr)/(ppv + tpr))

        
        val_dict['metrics']['tpr'] = np.mean(tpr) * 100
        val_dict['metrics']['tpr_std'] = np.std(tpr) * 100
        val_dict['metrics']['tnr'] = np.mean(tnr) * 100
        val_dict['metrics']['tnr_std'] = np.std(tnr) * 100
        val_dict['metrics']['ppv'] = np.mean(ppv) * 100
        val_dict['metrics']['ppv_std'] = np.std(ppv) * 100
        val_dict['metrics']['f1'] = np.mean(f1) * 100
        val_dict['metrics']['f1_std'] = np.std(f1) * 100


        metrics_stats.append(val_dict)

# Filename is complete path to file plus name and .txt extension
file = open(os.path.join(current_dir, 'trainer', 'logs', 'model_metrics.txt'), 'w')
file.write(metrics_stats)
file.close()


