#!/usr/bin/env python -W ignore::DataConversionWarning
import configparser
import json
import os
from urllib.parse import urlparse
from pymongo import MongoClient
from engine import training_dataframe, obtain_features, user_to_binary, ml_engine_training, save_scaling


# ---------------------------------------------------------------
# Some Configuration 
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
basedir = os.path.abspath(os.path.dirname(__file__))
checkpoint_path = os.path.join(basedir, 'checkpoints')
split = 1
# model = 'logRegr'
model = 'SVC'
# --------------------------------------------------------------

df = training_dataframe(mongodb_uri=MONGO_URI, split=split)
users = df['user_email'].unique()

# All the checkpoint sto be stored in checkpoints path
os.chdir(checkpoint_path)
for user in users:
    data = user_to_binary(df, user)
    # Aplicamos estandarización. Se guardará un fichero de estandarización en la carpeta checkpoints
    data = save_scaling(data)
    X_train, X_test, Y_train, Y_test = obtain_features(dataframe=data, upsampling=True)

    ml_engine_training(X_train, X_test, Y_train, Y_test, user, model=model)
    print('Training for user ', user, ' finished!')
os.chdir(basedir)
