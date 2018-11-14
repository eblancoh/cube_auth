# usr/bin/env python

import configparser
import os

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

basedir = os.path.abspath(os.path.dirname(__file__))

# Cargamos la configuración
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', '..', 'config', 'cfg.ini'))

no_feat = cfg.getint('deep_model', 'NO_FEATURES')
no_solvers = cfg.getint('deep_model', 'NO_SOLVERS')
num_lstm_units = cfg.getint('deep_model', 'NUM_LSTM_UNITS')
num_hidden_units_1 = cfg.getint('deep_model', 'NUM_HIDDEN_UNITS_1')
num_hidden_units_2 = cfg.getint('deep_model', 'NUM_HIDDEN_UNITS_2')
pad = cfg.getint('training', 'NUM_MOVES')


def lstm_rubik_model():
    """
    This function provides an architecture that takes no_feat different sequences.
    Each sequence is processed in a single LSTM branch.
    Finally, all previous branches are merged in a single one for further
    dense layers concatenation. The output layer provides a prediction on
    the label.
    :param : type=. .
    :param : type=. .
    :return: compiled model
    """

    x = list()
    for i in range(no_feat):
        x.append(Input(shape=(pad, 1)))


    # Create the branches of the model with LSTM layers
    # Let's define a number of LSTM units per layer.
    # This parameter could be modified
    # num_lstm_units = 30

    layer_branches = list()
    for k in x:
        lstm_1 = LSTM(units=num_lstm_units, return_sequences=True)(k)
        lstm_2 = LSTM(units=num_lstm_units, return_sequences=True)(lstm_1)
        lstm_3 = LSTM(units=num_lstm_units)(lstm_2)
        batch_norm_4 = BatchNormalization()(lstm_3)
        layer_branches.append([lstm_1, lstm_2, lstm_3, batch_norm_4])

    # Merge all previous branches in a single one.
    # We use keras.layers.merge.concatenate
    batch_norm = list()
    for k in layer_branches:
        batch_norm.append(k[-1])

    merge_layer = concatenate(batch_norm)

    # The model continues with dense hidden layers after the merging layer
    # Modify this parameter if necessary
    # num_hidden_units_1 = 100
    # First dense hidden layer after merge
    hidden_1 = Dense(units=num_hidden_units_1, activation='relu')(merge_layer)
    # Dropout layer after first hidden layer
    dropout_1 = Dropout(rate=0.10)(hidden_1)
    # Modify this parameter if necessary
    # num_hidden_units_2 = 50
    # Second dense hidden layer after merge
    hidden_2 = Dense(units=num_hidden_units_2, activation='relu')(dropout_1)
    # Dropout layer after second hidden layer
    dropout_2 = Dropout(rate=0.10)(hidden_2)
    # Output layer definition
    output = Dense(units=no_solvers, activation='softmax')(dropout_2)

    # The model is built according to architecture depicted above depending on the approach
    # The input to model is all the inputs to the model to be considered
    # The outputs are the labels obtained by the model given a rubik resolution
    # sequence.
    model = Model(inputs=x, outputs=output)

    return model
