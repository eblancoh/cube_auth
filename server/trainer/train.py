# usr/bin/env python

import configparser
import os

from trainer.input_builder import input_builder_deep
from trainer.model_builder import model_compiler, model_fitter, log_training, graph_model
from trainer.nn_lstm import lstm_rubik_model

# To silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))

# Cargando del fichero de configuración
config = configparser.ConfigParser()
config.read(os.path.join(current_dir, '..', '..', 'config', 'cfg.ini'))
num_moves = config.getint('training', 'NUM_MOVES')
epochs = config.getint('training', 'TRAINING_EPOCHS')
validation_split = config.getfloat('training', 'VALIDATION_SPLIT')
batch_size = config.getint('training', 'BATCH_SIZE')
save_graph = config.getboolean('training', 'SAVE_GRAPH')
use_logging = config.getboolean('training', 'USE_LOGGING')


def train_nn(pad, epochs, val_split, batch_size, use_logging, save_graph):
    """

    :param pad:
    :param epochs:
    :param use_logging:
    :param save_graph:
    :return:
    """

    print('Database connection committed. \nData loading for training started')
    # Loading data for training
    features, labels = input_builder_deep(pad=pad)

    # Build the model according to the architecture specified
    model = lstm_rubik_model(no_feat=features.__len__(), pad=num_moves, no_solvers=labels.shape[1])

    # The model shall be compiled before fitting process starts
    model_compiler(model=model, loss="categorical_crossentropy",
                   optimizer="adam", metrics=["accuracy"])

    print("Cube Auth Model training starts [!]")

    os.chdir(os.path.join(current_dir, "..", "checkpoints"))
    # We fit the model to our data-set (i.e., our state and our labels )
    callback = model_fitter(model=model, inputs=features, labels=labels, epochs=epochs,
                            validation_split=val_split, batch_size=batch_size, verbose=1)
    os.chdir(current_dir)

    print("Cube Auth Model training finished. Thanks for waiting [!]")

    # Logging for training and model graph representation
    if use_logging:
        if not os.path.exists(os.path.join(current_dir, "logs")):
            os.makedirs(os.path.join(current_dir, "logs"))
        # Training evolution is logged is use_logging= True
        log_training(callback, log_name="training_log",
                     filepath=os.path.join(current_dir, "logs"))
    if save_graph:
        if not os.path.exists(os.path.join(current_dir, "graphs")):
            os.makedirs(os.path.join(current_dir, "graphs"))
        # If save_graph=True, save the graph and summary
        # of our model and save it to graph_dir
        graph_model(model=model, graph_name='graph',
                    filepath=os.path.join(current_dir, "graphs"))
