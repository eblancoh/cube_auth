# usr/bin/env python

import json
import os
import numpy as np
import datetime
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf


# The sensitivity tells us the probability that we authenticate a user, 
# being the correct user. It is thus a measure 
# of how good we are at correctly authenticating a user.
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# The specificity tells us the probability that we do not allow a user authentication, 
# being the wrong user. It is thus a measure 
# of how good we are at correctly blocking a user authentication.
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def model_compiler(model, loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy', sensitivity, specificity, precision, recall]):
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
    ADAGRAD: optimizer with parameter-specific learning rates, which are adapted 
    relative to how frequently a parameter gets updated during training. 
    The more updates a parameter receives, the smaller the updates.
    ADAM: Adaptive Moment Estimation (ADAM) that also uses adaptive learning rates.

    You can specify the name of the loss function to use to the compile
    function by the loss argument. Some common examples include:

    'mse': for mean squared error.
    'binary_crossentropy': for binary logarithmic loss (logloss).
    'categorical_crossentropy': for multi-class logarithmic loss (logloss).
    """

    # The model is compiled with specified loss, optimizer and metrics
    # For a multi-class classification problem
    try:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    except BaseException:
        # If the above failed for some reason, simply
        print("Failed to compile provided model")


def model_loader(model, checkpoint_path):
    """

    :param model:
    :param checkpoint_path:
    :return:
    """

    try:
        print("Trying to restore last checkpoint ...")
        model.load_model(checkpoint_path, custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall})
        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored model from:", checkpoint_path)
    except BaseException:
        # If the above failed for some reason, simply
        print("Failed to restore model from: ", checkpoint_path)


def model_fitter(model, inputs, labels, email, epochs, validation_split, batch_size, class_weight, verbose):

    """
    This function fits tha created model following next inputs criteria
    :param model: compiled model
    :param inputs: independent variables to fit the model
    :param labels: labels for classification. One-hot encoded format.
    :param email: type=str. Email of the user to save the model
    :param epochs: type=int. Number of epochs to train the model.
    :param validation_split: type=float. Fraction of the dataset to be used
    for validation during training [0,1].
    :param batch_size: type=int. Size of the batch to use during training.
    :param verbose: type=bool. 0 or 1.
    :param checkpoint_base_dir: type=string. Path of the checkpoints to be stored.
    :return: fit_callback
    """
    # The checkpoint to be saved is indicated once every epoch is finished during training

    if not 'weights.best.' + email + '.h5' in os.listdir(os.getcwd()):
        checkpoint = ModelCheckpoint(filepath='weights.best.' + email + '.h5',
                                     monitor='val_precision',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=1)
        
        # Metemos Early Stopping para que el entrenamiento no se alargue en exceso.
        callbacks_list = [EarlyStopping(monitor='val_precision', patience=5), checkpoint]
        # Si el directorio de checkpoints está vacío
        fit_callback = model.fit(x=[inputs[i] for i in range(len(inputs))], 
                                 y=labels,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 batch_size=batch_size,
                                 callbacks=callbacks_list,
                                 class_weight=class_weight,
                                 shuffle=True,
                                 verbose=verbose)
    else:
        # Si el directorio de checkpoints no está vacío
        # carga el modelo desde el checkpoint
        model = load_model(filepath='weights.best.' + email + '.h5', compile=True, custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall})

        print('Model Loaded from previous training for %s [!]', email)

        checkpoint = ModelCheckpoint(filepath='weights.best.' + email + '.h5',
                                     monitor='val_precision',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=1)

        # Metemos Early Stopping para que el entrenamiento no se alargue en exceso.
        callbacks_list = [EarlyStopping(monitor='val_precision', patience=5), checkpoint]

        fit_callback = model.fit(x=[inputs[i] for i in range(len(inputs))], 
                                 y=labels,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 batch_size=batch_size,
                                 callbacks=callbacks_list,
                                 class_weight=class_weight,
                                 shuffle=True,
                                 verbose=verbose)
    return fit_callback


def log_training(callback, email, log_name, filepath):
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
        file.write(' | user:' + email + ' | train timestamp:' + str(datetime.datetime.now()) + '\n')


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


def model_predict(model, inputs, verbose=0):
    """

    :param model:
    :param inputs:
    :param verbose:
    :return:
    """
    y_pred = model.predict(x=[inputs[0], inputs[1], inputs[2]], verbose=verbose)[0]

    return y_pred 
    # return y_pred * 100, np.argmax(y_pred)
