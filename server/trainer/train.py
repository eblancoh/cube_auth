# usr/bin/env python

import os
import numpy as np
import argparse
from keras.preprocessing import sequence
from nn_lstm import lstm_rubik_model
from model_builder import model_compiler, model_fit, log_training, graph_model
from input_builder import input_builder_deep
import configparser

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

    # Loading data for training
    features, labels = input_builder_deep(pad=pad)
   
    # Build the model according to the architecture specified
    model = lstm_rubik_model(no_feat=features.__len__(), pad=num_moves, no_solvers=labels.shape[1])

    # The model shall be compiled before fitting process starts
    model_compiler(model=model, loss="categorical_crossentropy",
                   optimizer="adam", metrics=["accuracy"])
    
    # Building model with Keras for training routine finished
    print("Cube Auth Model built and compiled [!]")

    os.chdir(os.path.join(current_dir, "..", "server", "checkpoints"))
    # We fit the model to our data-set (i.e., our state and our labels )
    callback = model_fit(model=model, inputs=features, labels=labels, epochs=epochs, 
    validation_split=val_split, batch_size=batch_size, verbose=1)
    os.chdir(current_dir)

    print("Cube Auth Model trained [!]")

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


train_nn(pad=num_moves, epochs=epochs, val_split=validation_split, 
batch_size=batch_size, save_graph=save_graph, use_logging=use_logging)

"""
if __name__ == "__main__":
    desc = "Model for Rubik's Cube solving sequence classification training using Keras with TensorFlow backend"

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--pad",
                        help="number of moves contained in the sequences for training", type=int)
    parser.add_argument("--epochs",
                        help="number of epochs to train the model", type=int)
    parser.add_argument("--save_graph", required=False,
                        help="save graph of the model to be trained", type=bool)
    parser.add_argument("--save_log", required=False,
                        help="save log during model training", type=bool)

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Retrieve data from arguments parsed in terminal.
    pad = args.pad
    epochs = args.epochs
    save_graph = args.save_graph
    use_logging = args.save_log

    # Launch train routine
    trainer(pad=num_moves, epochs=epochs, save_graph=save_graph, use_logging=use_logging)

# TODO: visualización de pesos de cada capa
# TODO: rutinas de ploteo de evolución del loss y de la acc

"""