# usr/bin/env python

import os
import itertools
from .datasetbuilder import feed_training_data
from .modelbuilder import model_compiler, model_fitter, log_training, graph_model
from .neuralnetwork_conv import rubik_deep_model
from pymongo import MongoClient
from urllib.parse import urlparse
import numpy as np
from sklearn.utils import class_weight as cw

MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://cubeauth:cubeauth1233211@ds143070.mlab.com:43070/cubeauth')

# To silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))


def train(email, ep, val_split, batch_size, logging, graph, first):
    """
    This function supports training routine
    :param email: email del usuario que lanza el entrenamiento
    :param ep: number of epochs
    :param val_split: validation split (percentage of training set)
    :param batch_size: size of the batch of sequences
    :param logging: boolean
    :param graph: boolean
    :param first: boolean
    :return: trained model. Checkpoint of the model. Training log. Graph of the model.
    """
    print('Data load for training in progress.')
    # Loading data for training
    # features, labels, mapper = feed_training_data(email)
    if first:
        features, labels, pad, delta_t_ref = feed_training_data(email, first=True)
    else:
        # Cargar required_movements y time_ref
        # Connection to MongoDB is established
        client = MongoClient(MONGO_URI)
        # Getting a Database and parsing the name of the database from the MONGO_URI
        o = urlparse(MONGO_URI).path[1::]
        db = client[o]

        # Querying in MongoDB and obtaining the result as variable
        users_collection = list(db.users.find())

        for key, group in itertools.groupby(users_collection, key=lambda x: (x['email'])):
            for item in list(group):
                if item['email'] == email:
                    pad = item['required_movements']
                    delta_t_ref = item['time_ref']
        
        features, labels= feed_training_data(email, first=False)[0:2]
    
    # aprendizaje from imbalanced data. How to Handle Imbalanced Data
    # Esta ratio se utilizará en el entrenamiento del modelo.

    # desired tradeoff between sensitivity and specificity
    t = 1.0

    # class weights can easily be incorporated into the loss by adding the 
    # following parameter to the fit function (assuming that 1 is the user doing the correct sequence):

    class_weight = {0: 1., 
                    1: t * (labels.__len__() - np.count_nonzero(labels))/np.count_nonzero(labels)}
                    # 1: 1}

    # "treat every instance of class 1 as N instances of class 0 times a correction factor" means 
    # that in your loss function you assign higher value to these instances. Hence, 
    # the loss becomes a weighted average, where the weight of each sample is specified
    #  by class_weight and its corresponding class.

    # Build the model according to the architecture specified
    model = rubik_deep_model(pad, yaw_pitch_roll=features[2])

    # The model shall be compiled before fitting process starts
    model_compiler(model=model, loss="binary_crossentropy",
                optimizer="adam")
                # , metrics=["accuracy"])

    print("Model compiled!\nCube Auth Model training in progress.")

    os.chdir(os.path.join(current_dir, "..", "checkpoints"))
    # We fit the model to our data-set (i.e., our state and our labels )
    callback = model_fitter(model=model, inputs=features, labels=labels, email=email, epochs=ep,
                            validation_split=val_split, batch_size=batch_size, class_weight=class_weight, verbose=1)
    os.chdir(current_dir)

    print("Cube Auth Model training finished!\nThanks for waiting ;D")
    
    # Logging for training and model graph representation
    if logging:
        if not os.path.exists(os.path.join(current_dir, "logs")):
            os.makedirs(os.path.join(current_dir, "logs"))
        # Training evolution is logged is use_logging= True
        log_training(callback, email=email, log_name="training_log",
                    filepath=os.path.join(current_dir, "logs"))
    if graph:
        if not os.path.exists(os.path.join(current_dir, "graphs")):
            os.makedirs(os.path.join(current_dir, "graphs"))
        # If save_graph=True, save the graph and summary
        # of our model and save it to graph_dir
        graph_model(model=model, graph_name='graph',
                    filepath=os.path.join(current_dir, "graphs"))
                    
    return pad, delta_t_ref


