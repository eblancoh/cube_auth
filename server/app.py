# usr/bin/env python

import os

import numpy as np
from flask import Flask
from flask import jsonify, request
from pandas.io.json import json_normalize
from server.trainer import feat_df_app
from server.trainer.model_builder import load_checkpoint, model_compiler, model_predict
from server.trainer.nn_lstm import RubikModel

# To silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application
app = Flask(__name__)

# The current file"s location is obtained using __file__
current_dir = os.path.dirname(os.path.realpath(__file__))

# Number of movements to process in order to predict the class.
num_mov = 15
# Number of features contained in the x input delivered by app_input
num_feat = 2
# Number of classes used during Model's training
num_classes = 3


def app_input(df, no_mov):
    """

    :param df:
    :param no_mov:
    :return:
    """
    # Extract the features from the pd.DataFrame
    seq1, seq2 = feat_df_app(df=df)
    if len(seq1) < no_mov:
        # If the number of movements for a sequence is smaller than num_moves:
        seq1 = np.pad(np.array(seq1), (no_mov - len(seq1), 0), mode="constant")
        seq2 = np.pad(np.array(seq2), (no_mov - len(seq2), 0), mode="constant")
    else:
        # On the contrary, the sequences are padded with the last no_mov elements
        seq1 = seq1[0:no_mov]
        seq2 = seq2[0:no_mov]

    # The inputs are reshaped to fit the model
    seq1 = seq1.reshape(1, no_mov, 1)
    seq2 = seq2.reshape(1, no_mov, 1)
    #seq1 = seq1.values.reshape(1, no_mov, 1)
    #seq2 = seq2.values.reshape(1, no_mov, 1)
    # The input to our model is built:
    x = seq1, seq2

    return x


def load_model():
    """

    :return:
    """
    # Define a global variable for model
    global model

    # Build the model according to the architecture specified:
    model, _ = RubikModel(num_feat=num_feat, num_classes=num_classes, pad_length=num_mov)
    # Load the checkpoint once the model is defined:
    load_checkpoint(model, checkpoint_name="weights.best.hdf5", filepath=os.path.join(current_dir, "checkpoints"))
    # The model shall be compiled after the checkpoint is loaded:
    model_compiler(model=model, loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


@app.route("/rubik/predict", methods=["POST"])
def predict(no_mov=15):
    """

    :param no_mov:
    :return:
    """
    auth = {"success": False}
    #auth = {"success": True}

    if request.method == "POST":

        req_data = request.get_json()
        # Load req_data as a .json file element and normalize for every nested element
        # Generate a pd.DataFrame to start input processing to model
        df = json_normalize(req_data.get("sequence"))

        # The input to our model is built:
        x = app_input(df=df, no_mov=no_mov)
        # The real label of the provided sequence is obtained:
        y = req_data.get("id")

        # The label after the sequence is obtained
        prob_y_pred, y_pred = model_predict(model=model, inputs=x, verbose=0)

        # The result of the prediction is attached. Match probability is provided
        auth["predictions"] = []

        r = {"label": int(y_pred), "probability": float(prob_y_pred[0][y_pred])}
        auth["predictions"].append(r)

        if y_pred == y and float(prob_y_pred[0][y_pred]) > 50.:
            # If the predicted label is the same as the provided with the json file:
            auth["success"] = True

    # Return in json format the response:
    return jsonify(auth)


if __name__ == "__main__":
    longstring = """
    
                                CUBE AUTH
    
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNs--/ymMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMm+--:::--/smMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMh/-:::::::::--:smMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMNs:-::::::::::::-.`.symMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMNd/ .--:::::::::-.``/hmmysydMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMd+::.`  `.--::::-.`.odmmmmmmmhsydMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMh/://///:.`  `.--``-smmmmmmmmmmmmmh/+mMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMNy////////////:.  `. ymmmmmmmmmmmmmms-  .MMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMNo://////////////-` `` `:sdmmmmmmmmds-`   .NMMMMMMNMMMMMMMMMMMMMM
MMMMMMMMMMMMMmh/`-////////////.``.-::-.` `/ymmmmd+.`     .NMmhyssyddmNMMMMMMMMMM
MMMMMMMMMMMd+/+:.  .://////:.``.-::::::--.` `/s+`       ``-/hmmmmmmmmdhdNMMMMMMM
MMMMMMMMMy/+ooooo+-` `.:/-``.-::::::::::::--` `.  ``.-:////-./ydmmmmmmmmdhhmMMMM
MMMMMMNs/+oooooooooo/-` -``-::::::::::::::-.    .////////////:-./ymmmmmho/. hMMM
MMMMmo/ooooooooooooo+:``.  `.-:::::::--....-----.`-:////////////:.`//-```.--MMMM
MMMh`/oooooooooooo+-`.+dmh+.  `--:.``.--:::::::::-.``-////////:-.` .`-:///:dMMMM
MMMM+:`-+ooooooo/.`-odmmmmmmy/` ```. `.-::::::::::::-.``-:-.```... `//////oMMMMM
MMMMNyho-.:+o+:``:ymmmmmmmmmmmds: +mh+.`..-::::::::--..``.``.--::. ://///:NMMMMM
MMMMMNyddy/.-. /hmmmmmmmmmmmmmmds`dmmmmy/.``.-:--..``./: `-:::::- ./////:hMMMMMM
MMMMMMNsddddo` /ymmmmmmmmmmmmd+. :mmmmmmmds:`.- `:+ydNN: -:::::-``////:-hMMMMMMM
MMMMMMMN/ohddo```.odmmmmmmmh/.`  ymmmmmmmmmmh .hNNNNNNy `::::::. .:.```.MMMMMMMM
MMMMMMMMNhs+sd+`-.``/ymmmy:``   .mmmmmmmmmmm/ oNNNNNNN- -:::--.` ``.---yMMMMMMMM
MMMMMMMMMNhdh++..--..`./-       -hmmmmmmmmmh``mNNNNNNo .--.` ` `-:::::/MMMMMMMMM
MMMMMMMMMMmydddo..----.``       ` -odmmmmmm: oNNNNNNd` ` `.-:: .:::::-mMMMMMMMMM
MMMMMMMMMMMmoyddh....---``     :o/-``:ymmmy``mNNds+.``.-:////``-::::-sMMMMMMMMMM
MMMMMMMMMMMMm:/ohy.:-..-..    `+ooo+/. .+y- :s/.``.` `//////: .:::--+MMMMMMMMMMM
MMMMMMMMMMMMMd//:/:`:::..``   -ooooooo+:. .` `.-:/:` ://///:``--.``.dMMMMMMMMMMM
MMMMMMMMMMMMMMd////-`-:::-`  `+ooooooooo+` -://///. ./////:-  `:oydhMMMMMMMMMMMM
MMMMMMMMMMMMMMMMho//-`/-::-  -oooooooooo: `://///: `:/::-`` -hmmmmyNMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMNy+./y+:-` ./oooooooo+` -//////. `-``:oh-`hmmmmhmMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMmsoyys:-s:`-/ooooo- `/////:-``:ohmNNy /mmmmdhMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMNhyy/ymmdo-`:+o+` :/:-.``  yNNNNNN-`hmdddNMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMm/mmmmmmh+... `.``.--- :NNNNNNs-/dmMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMdymmmmmmmmmo``.-::::-``hNNNNddhMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMsmmmmmmmmmm: .::::::. /NmddmNMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMhhmmmmmmmmy `-:::::-``hmmMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMNdhdmmmmm- .::::-/smMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMmhhmms `-::+yNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNdh. -ohNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMmmMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

                                CUBE AUTH


"""
    print(longstring)

    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    # The model is loaded outside the app.route
    load_model()
    # El modo debug de flask en app.run() da un ValueError a la hora de dar una etiqueta.
    # ValueError: tensor tensor("dense_3/softmax:0", shape=(?, 10), dtype=float32) is not an element of this graph.
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5001)
    # Flask server started
