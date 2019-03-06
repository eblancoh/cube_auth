#!/usr/bin/env python
import configparser
import json
import os
import sys
from urllib.parse import urlparse
from pymongo import MongoClient
import itertools
from pymongo import MongoClient
from urllib.parse import urlparse
import numpy as np
import sklearn
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from engine import training_dataframe, obtain_features, user_to_binary, save_scaling
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


# ---------------------------------------------------------------
# Some configuration
basedir = os.path.abspath(os.path.dirname(__file__))
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')
checkpoint_path = os.path.join(basedir, 'checkpoints')
split = 1
model = 'SVC'
# model = 'logRegr'

loops = 2
repeat = 100
# --------------------------------------------------------------

df = training_dataframe(mongodb_uri=MONGO_URI, split=split)
users = df['user_email'].unique()

# fix random seed for reproducibility
seed = 314159
np.random.seed(seed)
# Define n_splits-fold cross validation test harness
kfold = StratifiedKFold(n_splits=loops, shuffle=True, random_state=seed)

dataframe = pd.DataFrame(columns=['user', 'tn', 'fp', 'fn', 'tp', 'p', 'r', 'f1'])
i = 0
os.chdir(checkpoint_path)
for j in range(repeat):
    for user in users:
    # All the checkpoints to be stored in checkpoints path
    
        df = user_to_binary(df, user)
        df = save_scaling(df)
        X = df.drop(['user', 'user_email', 'cube_type', 'is_random'], axis=1)
        Y = df['user']

        if model == 'logRegr':
            filename = 'logRegr_' + user + '.sav'
        elif model == 'SVC':
            filename = 'SVC_' + user + '.sav'

        if not filename in os.listdir(os.getcwd()):
            sys.exit("No training has been launched before or this user does not exist in database")
        else:
            loaded_model = pickle.load(open(filename, 'rb'))

            for train, test in kfold.split(X, Y):
                # Selecciono las muestras de test para cada validación k-Fold
                y_true = np.array(Y.iloc[test])
                x_test = X.iloc[test]
            
                # Predecimos las etiquetas
                y_pred = loaded_model.predict(x_test)

                # In the binary case, we can extract true positives, etc as follows:
                #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                _conf = confusion_matrix(y_true, y_pred).ravel()

                p = _conf[3]/(_conf[3] + _conf[1])
                r = _conf[3]/(_conf[3] + _conf[2])
                f1 = 2*p*r/(p+r)

                # Append to the pandas dataframe
                dataframe.loc[i] = [user, _conf[0], _conf[1], _conf[2], _conf[3], p, r, f1]
                i+=1

os.chdir(basedir)

dataframe.to_excel(model + "_metrics.xlsx")

print(dataframe.groupby(dataframe.user)[['p']].mean())
print(dataframe.groupby(dataframe.user)[['p']].std())
print(dataframe.groupby(dataframe.user)[['r']].mean())
print(dataframe.groupby(dataframe.user)[['r']].std())
print(dataframe.groupby(dataframe.user)[['f1']].mean())
print(dataframe.groupby(dataframe.user)[['f1']].std())
