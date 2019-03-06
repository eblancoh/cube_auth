#usr/bin/env python

import json
import pandas as pd
from pandas.io.json import json_normalize
import os
import numpy as np
from keras.preprocessing import sequence
from keras.layers import Input
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
import glob


#################################################################################################################
# Sequence and labels processing routines
def json_handler(jsons_path, pad_length):
    """
    This function extracts the sequences of interest and process them to be
    used as input to CONV1D model alternative.
    :param jsons_path: type=string. Path to the json files where sequences are stored
    :param pad_length: type=int. Maximum padding length for every sequence.
    :return: the concatenation of the sequences extracted from json files
    """

    # Move to the json_path where the .json files to be read are located
    # No matter if we are training or evaluating the model
    os.chdir(jsons_path)

    _sequence1 = []  # Timestamps will be stored in this variable
    _sequence2 = []  # Sides moved sequence will be stored in this variable
    _sequence3 = []  # Turns sequence will be considered in this variable

    # Locate all .json files in folder and extract the information
    files = [f for f in os.listdir('.') if f.endswith('.json')]
    for file in files:
        # Read the .json file to format it correctly
        with open(file, 'r') as item:
            _json = item.read()
        item.close()

        # Replace '\n' with commas
        _json = _json.replace('\n', ',')
        # Open and close the string with brackets
        _json = '[' + _json[0:-1] + ']'

        # Load the string as a .json file element
        df = json.loads(_json)

        # Normalize the .json file to get all nested keys in 'data'
        df = json_normalize(df)

        # List of nested keys.
        # keys = df.keys()
        # Not really necessary
        # Data of interest can be accessed using df['keys[i]']

        # Searching only those with 'data.sync' == True for TURN OK
        turn_ok = df.loc[df['data.sync'] == True, :]

        # Following data can be stored after _turnOK is identified:
        ## Sequence of shifts
        ## Sequence of shifted faces
        ## Timestamps of every valid movement
        turn_shift = turn_ok['data.shift']

        side_moved = turn_ok['data.side']

        time_stamp = turn_ok['stamp']
        # To avoid timeStamp handling, format is specified to Pandas
        time_stamp = pd.to_datetime(time_stamp,
                                   format='%Y-%m-%dT%H:%M:%S.%fZ')

        # All data to be appended
        _sequence1.append(time_stamp)
        _sequence2.append(side_moved)
        _sequence3.append(turn_shift)


    # Frst sequence (with time information) shall be processed to be used.
    sequence1 = [(elem - elem[0]) / np.timedelta64(1, 's') for elem in _sequence1]

    # No modifications are required for _sequence2
    sequence2 = _sequence2

    # Possible values considered:
    # CW (TurnRight), CCW (TurnLeft), 2CW (Double TurnRight), 2CCW (Double TurnLeft)
    # Following relationship is established:
    # CW (TurnRight) == 0; 2CW (Double TurnRight) == 1;
    # CCW (TurnLeft) == 2; 2CCW (Double TurnLeft) == 3.
    _turn_mapping = {'CW': 0, '2CW': 1, 'CCW': 2, '2CCW': 3}
    # The self.sequence3 variable is mapped according to described criteria
    sequence3 = [elem.map(_turn_mapping) for elem in _sequence3]

    # Truncate and pad input sequences. Force formats also.
    sequence1 = sequence.pad_sequences(sequence1,
                                       maxlen=pad_length, dtype='float64')
    sequence2 = sequence.pad_sequences(sequence2,
                                       maxlen=pad_length, dtype='int64')
    sequence3 = sequence.pad_sequences(sequence3,
                                       maxlen=pad_length, dtype='int64')

    seq = np.concatenate((sequence1, sequence2, sequence3), axis = 1)

    return np.expand_dims(seq, axis=2) # reshape from seq.shape to (seq.shape, 1)


def label_handler(labels_path):
    """
    This function read a .csv file filled with integers, identified as labels
    and returns a matrix of one-hot encoded labels.
    :param labels_path: type=string. Route to labels .csv file.
    :return: labels in one-hot encoded format.
    """

    # Move to the json_path where the .json files to be read are located
    # No matter if we are training or evaluating the model
    os.chdir(labels_path)

    # Locate all .csv files in folder and extract the information
    file = [f for f in os.listdir('.') if f.endswith('.csv')]
    with open(file[0]) as f:
        line = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    labels = [np.int(x.strip()) for x in line]
    # Transform from integers to categorical
    labels = to_categorical(labels)

    return labels

#################################################################################################################
# Neural Network definition,
# model training and app routines
# checkpoint handling routines
def neural_network(num_labels, pad_length):
    """
    This function provides an architecture that takes 3 different sequences
    concatenated. This sequence is entered in a CONV1D architecture.
    The output layer provides a prediction on
    the label.
    :param num_labels: type=int. Number of different labels to be considered in our model.
    :param pad_length: type=int. Maximum length of the sequence to be analyzed.
    :return: compiled model
    """

    input = Input(shape=(3*pad_length, 1))

    # First Convolutional layer to be fed with input layer.
    # Not necessary to specify the number of input dimensions for deeper layers.
    conv1d_1 = Conv1D(filters=64, kernel_size=3, activation='relu',
                     strides=1, padding='same')(input)

    # Second Convolutional layer.
    conv1d_2 = Conv1D(filters=64, kernel_size=3, activation='relu',
                     strides=1, padding='same')(conv1d_1)

    # Pooling layer
    pool_1 = MaxPooling1D(pool_size=3, strides=None, padding='same')(conv1d_2)

    # New Convolutional layer
    conv1d_3 = Conv1D(filters=64, kernel_size=3, activation='relu',
                      strides=1, padding='same')(pool_1)

    # New Convolutional layer
    conv1d_4 = Conv1D(filters=64, kernel_size=3, activation='relu',
                      strides=1, padding='same')(conv1d_3)

    # Global average pooling operation for temporal data.
    GlobAvg_1 = GlobalAveragePooling1D()(conv1d_4)

    # Dense layer after Global average pooling operation
    # Fully-connected layer with a certain number hidden units
    dense_1 = Dense(units=50, activation='relu')(GlobAvg_1)

    # Dropout layer after a dense layer
    dropout_1 = Dropout(rate=0.2)(dense_1)

    # Fully-connected layer as an output layer for label prediction
    output = Dense(units=num_labels, activation='softmax')(dropout_1)

    # The model is built according to architecture depicted above depending on the approach
    # The input to model is all the inputs to the model to be considered
    # The outputs are the labels obtained by the model given a rubik resolution
    # sequence.
    model = Model(inputs=input, outputs=output)

    return model


def model_compiler(model, loss='categorical_crossentropy',
                   optimizer='rmsprop', metrics=['accuracy']):
    """
    This function compiles the returned model by neural_network function.
    It is mandatory to compile the model before fitting is started.
    :param model: get the model returned by neural_network function
    :param loss: type=string.
    :param optimizer: type=string
    :param metrics: type=string
    :return:

        Some popular gradient descent optimizers you might like to choose from include:

        SGD: stochastic gradient descent, with support for momentum.
        RMSprop: adaptive learning rate optimization method proposed by Geoff Hinton.
        ADAM: Adaptive Moment Estimation (ADAM) that also uses adaptive learning rates.

        You can specify the name of the loss function to use to the compile
        function by the loss argument. Some common examples include:

        'mse': for mean squared error.
        'binary_crossentropy': for binary logarithmic loss (logloss).
        'categorical_crossentropy': for multi-class logarithmic loss (logloss).
    """

    # The model is compiled with specified loss, optimizer and metrics
    # For a multi-class classification problem
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def save_checkpoint(filepath):
    """
    This function save the checkpoints of our model in .hdf5 format in a checkpoint/ folder.
    :param filepath: type=string. The path where the checkpoint to be saved us stored
    :return: information to be used during model.fit
    """
    # Move to filepath directory for checkpoints
    os.chdir(filepath)

    # The checkpoint is defined
    checkpoint = ModelCheckpoint(filepath='weights-conv1d-epoch-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    return [checkpoint]


def load_checkpoint(model, checkpoint_name, filepath):
    """
    This function loads all variables of a trained graph from a checkpoint.
    If the checkpoint does not exist, it is notified to the user.
    :param model: get the model returned by neural_network function
    :param checkpoint_name: type=string. Name of the checkpoint the user desires to load.
    :param filepath: type=string. Filepath to look for desired checkpoint to be restored.
    :return:
    """

    try:
        print("Trying to restore last checkpoint ...")

        model.load_weights(os.path.join(filepath, checkpoint_name))

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", os.path.join(filepath, checkpoint_name))
    except:
        # If the above failed for some reason, simply
        print("Failed to restore checkpoint: ", checkpoint_name, " from:", filepath)


def model_fit(model, inputs, labels, epochs, validation_split, batch_size, verbose, checkpoint_base_dir):
    """
    This function fits tha created model following next inputs criteria
    :param model: compiled model
    :param inputs: independent variables to fit the model
    :param labels: labels for classification. One-hot encoded format.
    :param epochs: type=int. Number of epochs to train the model.
    :param validation_split: type=float. Fraction of the dataset to be used
    for validation during training [0,1].
    :param batch_size: type=int. Size of the batch to use during training.
    :param verbose: type=bool. 0 or 1.
    :param checkpoint_base_dir: type=string. Path of the checkpoints to be stored.
    :return: fit_callback
    """
    # The checkpoint to be saved is indicated once every epoch is finished during training
    checkpoint = save_checkpoint(filepath=checkpoint_base_dir)

    fit_callback = model.fit(x=inputs, y=labels,
                             epochs=epochs,
                             validation_split=validation_split,
                             batch_size=batch_size,
                             callbacks=checkpoint,
                             verbose=verbose)

    return fit_callback


def model_evaluate(model, inputs, labels, verbose):
    """
    This function provides the evaluation of a certain model already trained
    :param model: model already trained and compiled after load_checkpoint has been executed
    :param inputs: inputs to evaluate
    :param labels: labels to compare with predicted labels by oour trained model
    :param verbose: type=bool. 0 or 1.
    :return: scores of the evaluation
    """
    scores = model.evaluate(x=inputs, y=labels, batch_size=5, verbose=verbose)
    print('Accuracy over validation set is: %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


def log_training(callback, filepath):
    """
    This function is aimed at logging the training history in a .txt file located in logs folder
    :param callback: fit callback received after our model has been trained
    :param filepath: type=string. Filepath where the log will be stored.
    :return: a log file is created in filepath. The content to this file is appended;
    overwriting has been avoided.
    """
    history_callback = callback.history

    with open(file=filepath, mode='a', buffering=1) as file:
        json.dump(history_callback, file)
        file.write('\n')


def graph_model(model, graph_name, filepath):
    """
    This function created a graph and a summary of the architecture of our model
    :param model: model returned y neural_network function
    :param graph_name: type=string. Name of the .png and .txt model graph and summary to be stored in filepath
    :param filepath: type=string. Route to store both .png and .txt files.
    :return: .png and .txt files in filepath.
    """
    plot_model(model, show_shapes=True, to_file=filepath + graph_name + '.png')

    with open(filepath + graph_name + '.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print('Graph of the current model saved in', filepath + graph_name + '.png')

#################################################################################################################
# Agent for training and app our model
class Agent:
    """
    This class implements the function for running train and app of
    a certain model to figure out who was responsible of generate a certain
    sequence.
    """
    def __init__(self, training=True, save_graph=True, use_logging=True):
        """

        :param training: type=bool. If True, training is conducted.
        :param save_graph: type=bool. If True, graph and summary of the model is stored.
        :param use_logging: type=bool. If True, A log for training is kept.
        """
        # Define __init__ variables for the Agent class.
        self.training = training
        self.save_graph = save_graph
        self.use_logging = use_logging

        # The base_path of the project is included
        self.base_path = '/home/eblancoh/PycharmProjects/CubeAuth/'

        # Paths of interest paths are also included as variables
        self.src_path = os.path.join(self.base_path, 'trainer/')
        self.checkpoint_base_dir = os.path.join(self.base_path, 'checkpoints/')
        self.logs_base_dir = os.path.join(self.base_path, 'logs/')
        self.graph_dir = os.path.join(self.base_path, 'graphs/')

        # Make sure this directories exist if they don't
        if not os.path.exists(self.checkpoint_base_dir):
            os.makedirs(self.checkpoint_base_dir)
        if not os.path.exists(self.logs_base_dir):
            os.makedirs(self.logs_base_dir)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        if self.training:
            # If the model is expected to be trained, stats and labels will be extracted from:
            self.jsons_path = os.path.join(self.base_path, 'json-repository/train/')
            self.labels_path = os.path.join(self.base_path, 'labels-repository/train/')
        else:
            # If the model is expected to be tested, stats and labels will be extracted from:
            self.jsons_path = os.path.join(self.base_path, 'json-repository/evaluation/')
            self.labels_path = os.path.join(self.base_path, 'labels-repository/evaluation/')

        # The pad_length for each sequence is defined hereafter
        self.pad_length = 100

    def run(self):
        """
        The agent runs when this function is called for both training and app purposes.
        :return: training or app routines
        """

        # In case we are training the model, following activities will be done
        if self.training:
            print ('Training of CONV1D model starting')

            # Get the states stored in jsons and the labels for each resolution
            self.state = json_handler(self.jsons_path, self.pad_length)
            self.labels = label_handler(self.labels_path)
            # Move to source path
            os.chdir(self.src_path)

            # Number of features we use to train the model
            self.num_features = self.state.__len__()

            # Assert the number of .json files is the same as the number of assigned labels
            #assert self.labels.shape[0] == self.state[0].shape[0]

            # Build the model according to the architecture specified
            self.model = neural_network(num_labels=self.labels.shape[1],
                                             pad_length=self.pad_length)

            if self.save_graph:
                # If self.save_graph=True, save the graph and summary
                # of our model and save it to graph_dir
                graph_model(model=self.model, graph_name='conv1d-model', filepath=self.graph_dir)

            # The model shall be compiled before fitting process is started
            model_compiler(model=self.model, loss='categorical_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])

            # We fit the model to our data-set (i.e.,our state and our labels )
            self.callback = model_fit(model=self.model, inputs=self.state, labels=self.labels,
                                      epochs=50,
                                      validation_split=0.2,
                                      batch_size=10,
                                      verbose=1,
                                      checkpoint_base_dir=self.checkpoint_base_dir)
            # We move again to source path
            os.chdir(self.src_path)

            if self.use_logging:
                # Training evolution is logged is self.use_logging= True
                log_training(self.callback, filepath=self.logs_base_dir + 'log.txt')

        # In case we are evaluating the model (self.training = False),
        # following steps will be taken
        else:
            print('Evaluation of CONV1D model starting')

            # Get the states stored in jsons and the labels of each resolution for evaluation
            self.state = json_handler(self.jsons_path, self.pad_length)
            self.labels = label_handler(self.labels_path)
            # Move to source path
            os.chdir(self.src_path)

            # Build the model according to the architecture specified.
            # Note the same architecture as during training is required to successfully
            # make use of the checkpoint to restore
            self.model = neural_network(num_labels=self.labels.shape[1],
                                             pad_length=self.pad_length)

            # List all .hdf5 files located in self.checkpoint_base_dir directory and
            # select the most recent one to be delivered as variable to load_checkpoint
            # only those with naming convention weights-mlp-epoch-... .hdf5 are loaded.
            self.checkpoint_name = max(glob.glob(self.checkpoint_base_dir + '*conv1d*'),
                                       key=os.path.getctime)

            # Load the checkpoint once the model is defined
            load_checkpoint(self.model, checkpoint_name=self.checkpoint_name,
                            filepath=self.checkpoint_base_dir)

            # The model shall be compiled before evaluation
            model_compiler(model=self.model, loss='categorical_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])

            # Evaluation of the model can now be conducted
            model_evaluate(model=self.model, inputs=self.state, labels=self.labels, verbose=1)

##############################################################################################################

agent = Agent(training=True, save_graph=True, use_logging=True)
agent.run()