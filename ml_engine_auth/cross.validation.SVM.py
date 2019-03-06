# usr/bin/env python
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
logs_path = os.path.join(basedir, 'logs')
split = 1

df = training_dataframe(mongodb_uri=MONGO_URI, split=split)
users = df['user_email'].unique()

for user in users:
  # All the checkpoints to be stored in checkpoints path
    
  df = user_to_binary(df, user)

  os.chdir(checkpoint_path)
  df = save_scaling(df)
  os.chdir(basedir)

  X_train, X_test, Y_train, Y_test = obtain_features(dataframe=df, upsampling=True)

  # https://towardsdatascience.com/predicting-stars-galaxies-quasars-with-random-forest-classifiers-in-python-edb127878e43
  # https://chrisalbon.com/machine_learning/model_evaluation/cross_validation_parameter_tuning_grid_search/

  from sklearn.svm import SVC
  model = SVC()

  from pprint import pprint
  pprint(model.get_params())

  parameter_candidates = [
    {'C':  [1, 10, 20, 30, 40 ,50, 60 ,70, 80, 90, 100], 
    'kernel': ['linear']
    },
    {'C': [1, 10, 20, 30, 40 ,50, 60 ,70, 80, 90, 100], 
    'gamma': [20., 10., 5., 1., 0.1, 0.01, 0.001, 0.0001], 
    'kernel': ['rbf']
    },
  ]

  from sklearn.model_selection import GridSearchCV
  # Conduct Grid Search To Find Parameters Producing Highest Score
  rf_random = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, verbose=2, n_jobs=-1)
  # Train the classifier on X_train and Y_train
  rf_random.fit(X_train, Y_train)
  # A huge bunch of stuff comes up. To obtain the best parameters, we call:
  pprint(rf_random.best_params_)

  os.chdir(logs_path)
  with open("svc_GridSearch.txt", "a") as myfile:
      myfile.write(str(rf_random.best_params_) + "\n")
  os.chdir(basedir)

  print('Best score for training_data:', rf_random.best_score_) 

