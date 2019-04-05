#!/usr/bin/env python -W ignore::DataConversionWarning
# This script is aimed at launching a training routine for all the users included in the database.
import configparser
import json
import os
from urllib.parse import urlparse
from pymongo import MongoClient
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from engine import training_dataframe, obtain_features, user_to_binary, ml_engine_training, save_scaling, load_scaling


# Some Configuration 
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
basedir = os.path.abspath(os.path.dirname(__file__))
checkpoint_path = os.path.join(basedir, 'checkpoints')

# Following model to be supported
model = 'svc'
# model = 'logRegr'

# Loading dataframe from database
df = training_dataframe(mongodb_uri=MONGO_URI)
# Users involved in the experiment so far
users = df['user_email'].unique()

# All the checkpoints to be stored in checkpoints path
os.chdir(checkpoint_path)
for user in users:
    # Clasificación binaria para cada usuario
    data = user_to_binary(df, user)
    
    # Realizamos la partición del dataset 
    X_train, X_test, Y_train, Y_test = obtain_features(dataframe=data)

    # Aplicamos estandarización. Se guardará un fichero de estandarización en la carpeta checkpoints
    X_train = save_scaling(X_train)
    # Normalizamos el test dataset de acuerdo al training dataset sobre el que se ha hecho oversampling
    X_test = load_scaling(X_test)

    print('Training for user ', user)
    # The training is launched for user
    ml_engine_training(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test, user=user, model=model)
    
os.chdir(basedir)
