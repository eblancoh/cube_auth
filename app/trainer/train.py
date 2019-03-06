# usr/bin/env python

import os
from .mongodb import feed_training_data
from .model_builder import model_compiler, model_fitter, log_training, graph_model
from .nn_conv import rubik_deep_model

# To silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))


def train(ep, val_split, batch_size, logging, graph):
    """

    :param ep:
    :param val_split:
    :param batch_size:
    :param logging:
    :param graph:
    :return:
    """
    print('Data load for training in progress.')
    # Loading data for training
    features, labels, mapper = feed_training_data()

    # Build the model according to the architecture specified
    model = rubik_deep_model(YawPitchRoll=features[2])

    # The model shall be compiled before fitting process starts
    model_compiler(model=model, loss="categorical_crossentropy",
                   optimizer="adam", metrics=["accuracy"])

    print("Model compiled!\nCube Auth Model training in progress.")

    os.chdir(os.path.join(current_dir, "..", "checkpoints"))
    # We fit the model to our data-set (i.e., our state and our labels )
    callback = model_fitter(model=model, inputs=features, labels=labels, epochs=ep,
                            validation_split=val_split, batch_size=batch_size, verbose=1)
    os.chdir(current_dir)

    print("Cube Auth Model training finished!\nThanks for waiting ;D")
    
    # Logging for training and model graph representation
    if logging:
        if not os.path.exists(os.path.join(current_dir, "logs")):
            os.makedirs(os.path.join(current_dir, "logs"))
        # Training evolution is logged is use_logging= True
        log_training(callback, log_name="training_log",
                     filepath=os.path.join(current_dir, "logs"))
    if graph:
        if not os.path.exists(os.path.join(current_dir, "graphs")):
            os.makedirs(os.path.join(current_dir, "graphs"))
        # If save_graph=True, save the graph and summary
        # of our model and save it to graph_dir
        graph_model(model=model, graph_name='graph',
                    filepath=os.path.join(current_dir, "graphs"))
