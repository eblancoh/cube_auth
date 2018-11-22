#!/usr/bin/env python

import configparser
import datetime
import os
import time

import jwt
from backup_db import sqlite3_backup, clean_data
from flask import Flask, request, jsonify
from flask_jwt_extended import (JWTManager, create_access_token, jwt_required, get_jwt_identity, get_raw_jwt)
from flask_marshmallow import Marshmallow
from keras import backend as K
from keras.models import load_model
from trainer import (train_lstm, input_lstm, model_builder)

basedir = os.path.abspath(os.path.dirname(__file__))
database_dir = os.path.join(basedir, 'database_dir')
db_backup_dir = os.path.join(basedir, 'database_backup')

if not os.path.isdir(database_dir):
    os.mkdir(database_dir)

# Initialize the SQLAlchemy and Marshmallow extensions, in that order.
app = Flask(__name__)

# Configuration Loading
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
secret_key = cfg.get('database', 'SECRET_KEY')
jwt_secret_key = cfg.get('database', 'JWT_SECRET_KEY')
num_moves = cfg.getint('training', 'NUM_MOVES')
epochs = cfg.getint('training', 'TRAINING_EPOCHS')
validation_split = cfg.getfloat('training', 'VALIDATION_SPLIT')
batch_size = cfg.getint('training', 'BATCH_SIZE')
save_graph = cfg.getboolean('training', 'SAVE_GRAPH')
use_logging = cfg.getboolean('training', 'USE_LOGGING')
no_solvers = cfg.getint('deep_model', 'NO_SOLVERS')
prob_threshold = cfg.getfloat('predict', 'PROB_THRESHOLD')
auth_trigger = cfg.getint('database', 'AUTH_THRESHOLD')
token_expires_in = cfg.getint('database', 'TOKEN_EXPIRES_IN')
security_margin = cfg.getint('database', 'SECURITY_MARGIN')

# Configuration loading for the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(database_dir, 'CubeAuth.sqlite')
app.config['SECRET_KEY'] = secret_key
app.config['JWT_SECRET_KEY'] = jwt_secret_key
# To turn off UserWarning: SQLALCHEMY_TRACK_MODIFICATIONS adds significant
# overhead and will be disabled by default in the future.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Add support of token blacklisting
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access']

# Imports of the declared models for Database
from models.users import Users
from models.movements import Movements
from models.test_movements import TestMovements
from models.revoked_tokens import RevokedTokenModel
from models import db

JWT = JWTManager(app)
ma = Marshmallow(app)

# Generate marshmallow Schemas from your models using ModelSchema.
class UsersSchema(ma.ModelSchema):
    class Meta:
        model = Users


class MovementsSchema(ma.ModelSchema):
    class Meta:
        model = Movements


class TestMovementsSchema(ma.ModelSchema):
    class Meta:
        model = TestMovements


class RevokedTokenModelSchema(ma.ModelSchema):
    class Meta:
        model = RevokedTokenModel


users_schema = UsersSchema(many=True)
movements_schema = MovementsSchema(many=True)
test_movements_schema = TestMovementsSchema(many=True)
revoked_token_model_schema = RevokedTokenModelSchema(many=True)


#  Support of token blacklisting. Auxiliary function
@JWT.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token['jti']
    return RevokedTokenModel.is_jti_blacklisted(jti)

# Endpoints
#########################################################################
# Endpoint for token revoke
@app.route("/api/token_revoke", methods=["POST"])
@jwt_required
def token_revoke():
    jti = get_raw_jwt()['jti']

    try:
        revoked_token = RevokedTokenModel(jti=jti)
        revoked_token.add()

        ret = {
            'msg': 'Session token has been revoked'
        }
        return jsonify(ret)
    except:
        ret = {
            'msg': 'Something went wrong'
        }
        return jsonify(ret), 500


# Endpoint for authentication
@app.route("/api/authentication", methods=["POST"])
def authentication():
    if not request.is_json:
        return jsonify({"msg": "Missing json in request"}), 400

    # set the expiration period for new session tokens
    expires_in = datetime.timedelta(days=token_expires_in)
    expires_on = datetime.datetime.now() + expires_in

    # check if user is already in database
    username = request.json['username']

    # if the user does not exist in database:
    if Users.query.filter(Users.user == username).first() is None:

        # include new user in database
        new_user = Users(user=username)
        new_user.save_to_db()

        # generate new token
        token = create_access_token(identity=username, expires_delta=expires_in)

        Users.query.filter(Users.user == username).update({"session_token": token})
        Users.query.filter(Users.user == username).update({"session_token_exp": expires_on})

        # update information in database
        db.session.commit()

        ret = {
            'msg': 'KO',
            'session_token': token
        }

        return jsonify(ret)

    # if the user exists in database:
    else:

        # When client sends an expired JWT in my REST API, using decode_token from flask-extended-token
        # would meake impossible to process as "signature has expired".
        # That's why we use pyjwt with command jwt.decode(jwt_payload, 'secret', options={'verify_exp': False})
        jti = jwt.decode(Users.query.filter(Users.user == username).first().session_token,
                         jwt_secret_key, options={'verify_exp': False})

        # Defino un booleano que me indique si el token está expirado incluso dentro del límite de seguridad
        bool_expired = Users.query.filter(
            Users.user == username).first().session_token_exp < datetime.datetime.now() - datetime.timedelta(
            minutes=security_margin)
        # Defino un booleano que me indique si el token está en la blacklist
        bool_blacklist = check_if_token_in_blacklist(jti)

        # Si está expirado o en la blacklist:
        if bool_expired or bool_blacklist:
            # generate new token
            token = create_access_token(identity=username, expires_delta=expires_in)

            Users.query.filter(Users.user == username).update({"session_token": token})
            Users.query.filter(Users.user == username).update({"session_token_exp": expires_on})
            # update information in database
            db.session.commit()

            # Si el número de soluciones es menor al número mínimo de resoluciones para poder autenticar
            if Users.query.filter(Users.user == username).first().solutions < auth_trigger:
                ret = {
                    'msg': 'KO',
                    'session_token': token
                }
                return jsonify(ret)
            # Por el contrario, devolvemos un OK con el token
            else:
                ret = {
                    'msg': 'OK',
                    'session_token': token
                }
                return jsonify(ret)
        # Si el token no es inválido por fecha o está revocado
        else:
            # if token is still valid, then no token generation is necessary
            token = Users.query.filter(Users.user == username).first().session_token

            # Si el número de soluciones es menor al número mínimo de resoluciones para poder autenticar
            if Users.query.filter(Users.user == username).first().solutions < auth_trigger:
                ret = {
                    'msg': 'KO',
                    'session_token': token
                }
                return jsonify(ret)
            # Por el contrario, devolvemos un OK con el token
            else:
                ret = {
                    'msg': 'OK',
                    'session_token': token
                }
                return jsonify(ret)


# Enpoint for identification of user based on Authorization Access Token
@app.route('/api/protected', methods=['GET'])
@jwt_required
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


# Endpoint to increase in one the number of Users.solutions before a sequence
@app.route("/api/new_sequence_training", methods=["GET"])
# This Endpoint shall be called before sending a new set of moves through /api/move_training
# and /api/entire_sequence
@jwt_required
def new_sequence_training():
    username = get_jwt_identity()

    Users.query.filter(Users.user == username).update({"solutions": Users.solutions + 1})
    db.session.commit()
    ret = {
        "msg": "Sequence solution increased + 1 for User {}".format(username),
        'status': 'OK'
    }
    return jsonify(ret)


# Endpoint to update the value to include in of Movements.is_random for an entire sequence
@app.route("/api/is_random", methods=["POST"])
# This Endpoint shall be called before sending a new set of moves through /api/move_training
# and /api/entire_sequence
@jwt_required
def is_random():
    username = get_jwt_identity()

    req_data = request.get_json()

    random = req_data["random"]

    Users.query.filter(Users.user == username).update({"is_random": random})
    db.session.commit()

    if random:
        ret = {
            "msg": "Movement belongs to sequence from random start of the cube",
            'random': True
        }
    else:
        ret = {
            "msg": "Movement belongs to sequence from solved cube",
            'random': False
        }
    return jsonify(ret)


# Endpoint to enter new movements from the Rubik's Cube
@app.route("/api/move_training", methods=["POST"])
@jwt_required
def add_train_movement():
    username = get_jwt_identity()

    req_data = request.get_json()

    try:
        var = req_data['data']['position']
        # si existe el key anidado de posicionamiento pero viene como null
        if req_data['data']['position'] is None:
            cube_type = "generic"
            data_x = None
            data_y = None
            data_z = None
        # si existe y viene completado con la información
        else:
            cube_type = "11paths"
            data_x = req_data['data']['position']['X']
            data_y = req_data['data']['position']['Y']
            data_z = req_data['data']['position']['Z']
    # En el caso de que directamente no exista la información de posicionamiento
    except KeyError:
        data_x = None
        data_y = None
        data_z = None
        cube_type = "generic"

    solution = Users.query.filter(Users.user == username).first().solutions
    user_id = Users.query.filter(Users.user == username).first().id
    index = req_data['index']
    stamp = req_data['stamp']
    src = req_data['src']
    event = req_data['event']
    data_code = req_data['data']['code']
    data_order = req_data['data']['order']
    data_sync = req_data['data']['sync']
    data_lost = req_data['data']['lost']
    data_side = req_data['data']['side']
    data_shift = req_data['data']['shift']
    data_dbm = req_data['data']['dbm']
    data_rssi = req_data['data']['rssi']
    is_random = Users.query.filter(Users.user == username).first().is_random

    new_move = Movements(user=username, user_id=user_id, solution=solution, index=index, stamp=stamp,
                         src=src, event=event, data_code=data_code, data_order=data_order,
                         data_sync=data_sync, data_lost=data_lost, data_side=data_side,
                         data_shift=data_shift, data_dbm=data_dbm, data_rssi=data_rssi,
                         data_x=data_x, data_y=data_y, data_z=data_z,
                         is_random=is_random, cube_type=cube_type)
    new_move.save_to_db()

    ret = {
        'msg': 'Movement successfully stored',
        "status": "OK"
    }
    return jsonify(ret)


# Endpoint to enter new entire sequences from a certain user
@app.route("/api/entire_sequence", methods=["POST"])
@jwt_required
def add_sequence():
    username = get_jwt_identity()

    req_data = request.get_json()['sequence']

    for item in req_data:
        try:
            var = item['data']['position']
            # si existe el key anidado de posicionamiento pero viene como null
            if item['data']['position'] is None:
                cube_type = "generic"
                data_x = None
                data_y = None
                data_z = None
            # si existe y viene completado con la información
            else:
                cube_type = "11paths"
                data_x = req_data['data']['position']['X']
                data_y = req_data['data']['position']['Y']
                data_z = req_data['data']['position']['Z']
        # En el caso de que directamente no exista la información de posicionamiento
        except KeyError:
            data_x = None
            data_y = None
            data_z = None
            cube_type = "generic"

        solution = Users.query.filter(Users.user == username).first().solutions
        user_id = Users.query.filter(Users.user == username).first().id
        index = item['index']
        stamp = item['stamp']
        src = item['src']
        event = item['event']
        data_code = item['data']['code']
        data_order = item['data']['order']
        data_sync = item['data']['sync']
        data_lost = item['data']['lost']
        data_side = item['data']['side']
        data_shift = item['data']['shift']
        data_dbm = item['data']['dbm']
        data_rssi = item['data']['rssi']
        is_random = Users.query.filter(Users.user == username).first().is_random

        new_move = Movements(user=username, user_id=user_id, solution=solution, index=index, stamp=stamp,
                             src=src, event=event, data_code=data_code, data_order=data_order,
                             data_sync=data_sync, data_lost=data_lost, data_side=data_side,
                             data_shift=data_shift, data_dbm=data_dbm, data_rssi=data_rssi,
                             data_x=data_x, data_y=data_y, data_z=data_z,
                             is_random=is_random, cube_type=cube_type)
        new_move.save_to_db()

    ret = {
        'msg': 'Sequence successfully stored',
        "status": "OK"
    }

    return jsonify(ret)


# Endpoint to check the number of solving sequences filtered by user
@app.route("/api/auth_threshold", methods=["GET"])
@jwt_required
def auth_threshold():
    username = get_jwt_identity()
    result = Users.query.filter(Users.user == username).first().solutions
    # threshold = Users.query.filter(Users.user == username).first().auth_threshold
    if result >= auth_trigger:
        ret = {
            'msg': 'Authentication attempt could be allowed',
            'status': 'END'
        }
        return jsonify(ret)
    else:
        ret = {
            'msg': 'Authentication attempt could not be allowed with {} pending solutions'.format(auth_trigger - result),
            'status': 'CONTINUE'
        }
        return jsonify(ret)


# Endpoint to launch training of the model once an stop is received
# and backup database after the training is completed
@app.route("/api/train_lstm", methods=["GET"])
@jwt_required
def train_deep(ep=epochs, val_split=validation_split, batch=batch_size,
               bool_graph=save_graph, bool_logging=use_logging):
    # This routine launches the training of the model after
    # a GET petition from Frontend

    # Training routine is launched. Checkpoint is stored in server/checkpoints folder
    train_lstm.train_lstm(ep=ep, val_split=val_split, batch_size=batch,
                          graph=bool_graph, logging=bool_logging)

    # Database Backup triggered after training is finished
    sqlite3_backup('CubeAuth.sqlite', db_backup_dir)

    # Delete backups older than NO_OF_DAYS days
    clean_data(db_backup_dir)

    # To delete the empty .sqlite database created after backup
    for file in os.listdir(basedir):
        if file.endswith(".sqlite"):
            os.remove(file)

    # Revocamos el token tras el entrenamiento del modelo.
    jti = get_raw_jwt()['jti']
    revoked_token = RevokedTokenModel(jti=jti)
    revoked_token.add()

    ret = {
        'msg': 'Cube auth Model successfully trained and database back-up completed',
        'status': 'OK'
    }

    return jsonify(ret)


# Endpoint to launch testing of the model
@app.route("/api/predict", methods=["POST"])
@jwt_required
def predict():
    username = get_jwt_identity()

    # Check if there is any checkpoint in checkpoints folder
    if len(os.listdir(os.path.join(basedir, 'checkpoints'))) == 0:
        ret = {
                'msg': "Consider training a model. No checkpoints found.",
                'status': 'KO'
        }
        return jsonify(ret)

    else:

        req_data = request.get_json()

        auth = {"success": False}

        if TestMovements.query.all().__len__() < num_moves:

            try:
                var = req_data['data']['position']
                # Si existe el key anidado de posicionamiento pero viene como null
                if req_data['data']['position'] is None:
                    cube_type = "generic"
                    data_x = None
                    data_y = None
                    data_z = None
                # si existe y viene completado con la información
                else:
                    cube_type = "11paths"
                    data_x = req_data['data']['position']['X']
                    data_y = req_data['data']['position']['Y']
                    data_z = req_data['data']['position']['Z']
            # En el caso de que directamente no exista la información de posicionamiento
            except KeyError:
                data_x = None
                data_y = None
                data_z = None
                cube_type = "generic"

            solution = Users.query.filter(Users.user == username).first().solutions
            user_id = Users.query.filter(Users.user == username).first().id
            index = req_data['index']
            stamp = req_data['stamp']
            src = req_data['src']
            event = req_data['event']
            data_code = req_data['data']['code']
            data_order = req_data['data']['order']
            data_sync = req_data['data']['sync']
            data_lost = req_data['data']['lost']
            data_side = req_data['data']['side']
            data_shift = req_data['data']['shift']
            data_dbm = req_data['data']['dbm']
            data_rssi = req_data['data']['rssi']
            is_random = Users.query.filter(Users.user == username).first().is_random

            new_move = TestMovements(user=username, user_id=user_id, solution=solution, index=index, stamp=stamp,
                                     src=src, event=event, data_code=data_code, data_order=data_order,
                                     data_sync=data_sync, data_lost=data_lost, data_side=data_side,
                                     data_shift=data_shift, data_dbm=data_dbm, data_rssi=data_rssi,
                                     data_x=data_x, data_y=data_y, data_z=data_z,
                                     is_random=is_random, cube_type=cube_type)

            new_move.save_to_db()

            # Return the movement, enter a new movement for db storing
            return jsonify(req_data)

        else:
            # Carga de las características de la secuencia
            features = input_lstm.input_to_test_lstm()

            # Model is loaded
            # TODO: ¿Cómo sacar fuera esto? tarda bastante
            start_time = time.time()

            K.clear_session()
            model = load_model(os.path.join(basedir, 'checkpoints', 'weights.best.h5'))

            print("--- Model loading took %s seconds ---" % (time.time() - start_time))

            # The label after the sequence is obtained
            probability, y_pred = model_builder.model_predict(model=model, inputs=features, verbose=0)

            # Obtenemos la etiqueta desde el token facilitado
            q = Users.query.filter(Users.user == username).first().id

            # The result of the prediction is attached. Match probability is provided
            auth["predict"] = []
            prob_y_pred = probability[0][y_pred]
            ret = {
                "label": int(y_pred),
                "probability": float(prob_y_pred)
            }
            auth["predict"].append(ret)

            if y_pred == q and prob_y_pred > prob_threshold:
                # If the predicted label is the same as the provided with the json file:
                auth["success"] = True

            # Delete TestMovements table after providing auth json once the model has been tested
            TestMovements.query.delete()
            # Commit deletion to database
            db.session.commit()

            # Revocamos el token tras la predicción de la etiqueta.
            jti = get_raw_jwt()['jti']
            revoked_token = RevokedTokenModel(jti=jti)
            revoked_token.add()

            # Return in json format the response:
            return jsonify(auth)


# Query Endpoints in Database
# Endpoint to show all the info from the Users database
@app.route("/api/query/all_users", methods=["GET"])
def get_user():
    all_users = Users.query.all()
    result = users_schema.dump(all_users)
    return jsonify(result.data)


# Endpoint to show all the info from the Movements database
@app.route("/api/query/all_movements", methods=["GET"])
def get_movements():
    all_movements = Movements.query.all()
    result = movements_schema.dump(all_movements)
    return jsonify(result.data)


# Endpoint to show all the info from the Movements database
@app.route("/api/query/test_movements", methods=["GET"])
@jwt_required
def get_test_movements():
    test_movements = TestMovements.query.all()
    result = test_movements_schema.dump(test_movements)
    return jsonify(result.data)


# Endpoint to get movements filtered by user
@app.route("/api/query/movements", methods=["GET"])
@jwt_required
def query_movements():
    username = get_jwt_identity()
    q = Movements.query.join(Users, Movements.parent).filter(Users.user == username)
    result = movements_schema.dump(q)
    return jsonify(result)


if __name__ == '__main__':
    db.create_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
