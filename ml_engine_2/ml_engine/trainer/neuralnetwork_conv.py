# usr/bin/env python

import configparser
import os
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

basedir = os.path.abspath(os.path.dirname(__file__))

# Configuration loading from config/cfg.ini
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', '..', 'config', 'cfg.ini'))
num_lstm_units = cfg.getint('deep_model', 'NUM_LSTM_UNITS')
conv2d_filters_1 = cfg.getint('deep_model', 'CONV2D_FILTERS_1')
conv2d_filters_2 = cfg.getint('deep_model', 'CONV2D_FILTERS_2')
hidden_conv2d_1_units = cfg.getint('deep_model', 'HIDDEN_CONV2D_1_UNITS')
hidden_conv2d_2_units = cfg.getint('deep_model', 'HIDDEN_CONV2D_2_UNITS')
num_hidden_units_1 = cfg.getint('deep_model', 'NUM_HIDDEN_UNITS_1')
num_hidden_units_2 = cfg.getint('deep_model', 'NUM_HIDDEN_UNITS_2')
num_hidden_units_3 = cfg.getint('deep_model', 'NUM_HIDDEN_UNITS_3')


def rubik_deep_model(pad, yaw_pitch_roll):

    # Branch for delta times processing
    input_dt = Input(shape=(pad, 1))
    lstm_dt_1 = LSTM(units=num_lstm_units, return_sequences=True)(input_dt)
    lstm_dt_2 = LSTM(units=num_lstm_units, return_sequences=True)(lstm_dt_1)
    lstm_dt_3 = LSTM(units=num_lstm_units)(lstm_dt_2)
    batch_norm_dt = BatchNormalization()(lstm_dt_3)

    # Branch for turns sequences processing
    input_turns = Input(shape=(pad, 1))
    lstm_turns_1 = LSTM(units=num_lstm_units, return_sequences=True)(input_turns)
    lstm_turns_2 = LSTM(units=num_lstm_units, return_sequences=True)(lstm_turns_1)
    lstm_turns_3 = LSTM(units=num_lstm_units)(lstm_turns_2)
    batch_norm_turns = BatchNormalization()(lstm_turns_3)

    # Branch for convolutional processing of Yaw-Pitch-Roll sequences
    ypr_shape = yaw_pitch_roll.shape
    input_ypr = Input(shape=(ypr_shape[1], ypr_shape[2], ypr_shape[3]))
    conv2d_1 = Conv2D(filters=conv2d_filters_1, kernel_size=3, padding='same', data_format='channels_last',
                      activation='relu')(input_ypr)
    maxpool2d_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_1)
    conv2d_2 = Conv2D(filters=conv2d_filters_2, kernel_size=3, padding='same', activation='relu')(maxpool2d_1)
    maxpool2d_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)
    flatten = Flatten(name='flatten')(maxpool2d_2)
    hidden_conv2d_1 = Dense(units=hidden_conv2d_1_units, activation='relu')(flatten)
    hidden_conv2d_2 = Dense(units=hidden_conv2d_2_units, activation='relu')(hidden_conv2d_1)
    batch_norm_conv = BatchNormalization()(hidden_conv2d_2)
    
    # Merge all previous branches in a single one.
    # We use keras.layers.merge.concatenate
    merge_layer = concatenate([batch_norm_dt, batch_norm_turns, batch_norm_conv])

    # The model continues with dense hidden layers after the merging layer
    # Modify this parameter if necessary
    # First dense hidden layer after merge
    hidden_1 = Dense(units=num_hidden_units_1, activation='relu')(merge_layer)
    # Dropout layer after first hidden layer
    dropout_1 = Dropout(rate=0.10)(hidden_1)
    # Modify this parameter if necessary
    # Second dense hidden layer after merge
    hidden_2 = Dense(units=num_hidden_units_2, activation='relu')(dropout_1)
    # Dropout layer after second hidden layer
    dropout_2 = Dropout(rate=0.10)(hidden_2)
    # Third dense hidden 
    hidden_3 = Dense(units=num_hidden_units_3, activation='relu')(dropout_2)
    # Dropout layer after second hidden layer
    dropout_3 = Dropout(rate=0.10)(hidden_3)
    # Output layer definition
    # output = Dense(units=no_solvers, activation='softmax')(dropout_2)
    output = Dense(units=1, kernel_initializer='normal', activation='sigmoid')(dropout_3)

    # The model is built according to architecture depicted above depending on the approach
    # The input to model is all the inputs to the model to be considered
    # The outputs are the labels obtained by the model given a rubik resolution
    # sequence.
    model = Model(inputs=[input_dt, input_turns, input_ypr], outputs=output)

    return model
