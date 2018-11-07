# usr/bin/env python

import json
import os
import numpy as np
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))

def model_compiler(model, loss='categorical_crossentropy',
                   optimizer='rmsprop', metrics=[metrics.categorical_accuracy]):
    """
    This function compiles the returned model by neural_network function.
    It is mandatory to compile the model before fitting is started.
    :param model: get the model returned by neural_network function
    :param loss: type=string.
    :param optimizer: type=string
    :param metrics: type=string
    :return: model compiled

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


def save_checkpoint():
    """
    This function save the checkpoints of our model in .hdf5 format in a checkpoint/ folder.
    :param filepath: type=string. The path where the checkpoint to be saved us stored
    :return: information to be used during model.fit
    """
    # The checkpoint is defined
    checkpoint = ModelCheckpoint(filepath='weights.best.hdf5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)

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


def model_fit(model, inputs, labels, epochs, validation_split, batch_size, verbose):

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
    #checkpoint = save_checkpoint(filepath=checkpoint_base_dir)
    checkpoint = save_checkpoint()



    fit_callback = model.fit(x=[inputs[i] for i in range(len(inputs))], y=labels,
                             epochs=epochs,
                             validation_split=validation_split,
                             batch_size=batch_size,
                             callbacks=checkpoint,
                             shuffle=True,
                             verbose=verbose)
    return fit_callback


def log_training(callback, log_name, filepath):
    """
    This function is aimed at logging the training history in a .txt file located in logs folder
    :param log_name:
    :param callback: fit callback received after our model has been trained
    :param filepath: type=string. Filepath where the log will be stored.
    :return: a log file is created in filepath. The content to this file is appended;
    overwriting has been avoided.
    """
    history_callback = callback.history

    with open(file=os.path.join(filepath, log_name + ".txt"), mode='a', buffering=1) as file:
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

    plot_model(model, show_shapes=True, to_file=os.path.join(filepath, graph_name + '.png'))

    with open(os.path.join(filepath, graph_name + '.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print('Graph of the current model saved in: ', os.path.join(filepath, graph_name + '.png'))


def model_evaluate(model, inputs, labels, verbose):
    """
    This function provides the evaluation of a certain model already trained
    :param model: model already trained and compiled after load_checkpoint has been executed
    :param inputs: inputs to evaluate
    :param labels: labels to compare with predicted labels by oour trained model
    :param verbose: type=bool. 0 or 1.
    :return: scores of the evaluation
    """
    scores = model.evaluate(x=[inputs[i] for i in range(len(inputs))], y=labels, batch_size=1, verbose=verbose)
    print('Accuracy over validation set is: %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


def model_predict(model, inputs, verbose=0):
    """

    :param model:
    :param inputs:
    :param verbose:
    :return:
    """
    # y_pred = model.predict(x=[inputs[i] for i in range(len(inputs))], verbose=verbose)
    y_pred = model.predict(x=[inputs[i] for i in range(len(inputs))], verbose=verbose)

    return y_pred * 100, np.argmax(y_pred)
