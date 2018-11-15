#!/usr/bin/env python

import configparser
import datetime
import os
import time

from backup_db import sqlite3_backup, clean_data
from flask import Flask, request, jsonify
from flask_jwt_extended import (JWTManager, create_access_token,
                                jwt_required, get_jwt_identity)
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from keras import backend as K
from keras.models import load_model
from sqlalchemy import (Column, String, Boolean, Integer,
                        Float, ForeignKey, DateTime)
from sqlalchemy.orm import relationship
from trainer import (train_lstm, input_lstm, model_builder)

basedir = os.path.abspath(os.path.dirname(__file__))
database_dir = os.path.join(basedir, 'database_dir')
db_backup_dir = os.path.join(basedir, 'database_backup')

if not os.path.isdir(database_dir):
    os.mkdir(database_dir)

# Carga de la configuración
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

# Initialize the SQLAlchemy and Marshmallow extensions, in that order.
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(database_dir, 'CubeAuth.sqlite')
app.config['SECRET_KEY'] = secret_key
app.config['JWT_SECRET_KEY'] = jwt_secret_key
# To turn off UserWarning: SQLALCHEMY_TRACK_MODIFICATIONS adds significant
# overhead and will be disabled by default in the future.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

JWT = JWTManager(app)

db = SQLAlchemy(app)
ma = Marshmallow(app)


# Declare the models of the DataBase
# Table Users in linked to Movements in a parent -> child relationship
class Users(db.Model):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user = Column(String, unique=True)
    session_token = Column(String)
    session_token_exp = Column(DateTime)
    session_token_usage = Column(Integer, default=0)
    solutions = Column(Integer, default=0)
    auth_threshold = Column(Integer, default=10)
    is_random = Column(Boolean)
    child = relationship("Movements", back_populates="parent")

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def __init__(self, user):
        self.user = user


class Movements(db.Model):
    __tablename__ = 'movements'

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey('users.user'))
    user_id = Column(Integer)
    solution = Column(Integer)
    index = Column(Integer)
    stamp = Column(String)
    src = Column(String)
    event = Column(String)
    data_code = Column(String)
    data_order = Column(Integer)
    data_sync = Column(Boolean)
    data_lost = Column(Integer)
    data_side = Column(Integer)
    data_shift = Column(String)
    data_dbm = Column(Float)
    data_rssi = Column(Float)
    data_x = Column(Float)
    data_y = Column(Float)
    data_z = Column(Float)
    is_random = Column(Boolean)
    cube_type = Column(String)
    parent = relationship(Users, back_populates="child")

    def __init__(self,
                 user, user_id, solution, index, stamp, src, event, data_code,
                 data_order, data_sync, data_lost, data_side, data_shift,
                 data_dbm, data_rssi, data_x, data_y, data_z, is_random,
                 cube_type):
        self.user = user
        self.user_id = user_id
        self.solution = solution
        self.index = index
        self.stamp = stamp
        self.src = src
        self.event = event
        self.data_code = data_code
        self.data_order = data_order
        self.data_sync = data_sync
        self.data_lost = data_lost
        self.data_side = data_side
        self.data_shift = data_shift
        self.data_dbm = data_dbm
        self.data_rssi = data_rssi
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.is_random = is_random
        self.cube_type = cube_type

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()


class TestMovements(db.Model):
    __tablename__ = 'test_movements'

    id = Column(Integer, primary_key=True)
    user = Column(String)
    user_id = Column(Integer)
    solution = Column(Integer)
    index = Column(Integer)
    stamp = Column(String)
    src = Column(String)
    event = Column(String)
    data_code = Column(String)
    data_order = Column(Integer)
    data_sync = Column(Boolean)
    data_lost = Column(Integer)
    data_side = Column(Integer)
    data_shift = Column(String)
    data_dbm = Column(Float)
    data_rssi = Column(Float)
    data_x = Column(Float)
    data_y = Column(Float)
    data_z = Column(Float)
    is_random = Column(Boolean)
    cube_type = Column(String)

    def __init__(self,
                 user, user_id, solution, index, stamp, src, event, data_code,
                 data_order, data_sync, data_lost, data_side, data_shift,
                 data_dbm, data_rssi, data_x, data_y, data_z, is_random,
                 cube_type):
        self.user = user
        self.user_id = user_id
        self.solution = solution
        self.index = index
        self.stamp = stamp
        self.src = src
        self.event = event
        self.data_code = data_code
        self.data_order = data_order
        self.data_sync = data_sync
        self.data_lost = data_lost
        self.data_side = data_side
        self.data_shift = data_shift
        self.data_dbm = data_dbm
        self.data_rssi = data_rssi
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.is_random = is_random
        self.cube_type = cube_type

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()


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


users_schema = UsersSchema(many=True)
movements_schema = MovementsSchema(many=True)
test_movements_schema = TestMovementsSchema(many=True)


# Endpoint for authentication
@app.route("/api/authentication", methods=["POST"])
def authentication():
    if not request.is_json:
        return jsonify({"msg": "Missing json in request"}), 400

    # check if user is already in database
    username = request.json['username']

    # set the expiration period of the session token
    expires_in = datetime.timedelta(days=1)
    expires_on = datetime.datetime.now() + expires_in

    # if the user exists
    if Users.query.filter(Users.user == username).first() is not None:
        # if the token is still valid in time
        if Users.query.filter(Users.user == username).first().session_token_exp < datetime.datetime.now():
            # generate new token
            token = create_access_token(identity=username, expires_delta=expires_in)
            # refresh_token = create_refresh_token(identity=username, expires_delta=expires_refresh)

            Users.query.filter(Users.user == username).update({"session_token": token})
            Users.query.filter(Users.user == username).update({"session_token_exp": expires_on})
            # update information in database
            db.session.commit()

            ret = {
                'msg': 'Existing user in database. New token generated',
                'session_token': token
            }
            return jsonify(ret)
        else:
            # if token is still valid, then no token generation is necessary
            token = Users.query.filter(Users.user == username).first().session_token
            ret = {
                'msg': 'Existing user in database. Existing token still valid',
                'session_token': token
            }
            return jsonify(ret)

    try:  # if the user is not included in the database

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
            'msg': 'User {} was created'.format(request.json['username']),
            'session_token': token
        }
        return jsonify(ret)
    except Exception:
        return {'msg': 'Something went wrong'}, 500


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


# Endpoint to check the number of solving sequences filtered by user
@app.route("/api/auth_threshold", methods=["GET"])
@jwt_required
def auth_threshold():
    username = get_jwt_identity()
    result = Users.query.filter(Users.user == username).first().solutions
    threshold = Users.query.filter(Users.user == username).first().auth_threshold
    if result >= threshold:
        ret = {
            'msg': 'Authentication attempt allowed',
            'status': 'OK'
        }
        return jsonify(ret)
    else:
        ret = {
            'msg': 'Authentication attempt not allowed with {} pending solutions'.format(threshold - result),
            'status': 'KO'
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

    ret = {
        'msg': 'Cube auth Model successfully trained and database back-up completed',
        'status': 'OK'
    }
    return jsonify(ret)


# Endpoint to launch testing of the model
# /api/load_model shall be called before
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

            if y_pred == q and prob_y_pred > 50.:
                # If the predicted label is the same as the provided with the json file:
                auth["success"] = True

            # Delete TestMovements table after providing auth json once the model has been tested
            TestMovements.query.delete()
            # Commit deletion to database
            db.session.commit()

            # Return in json format the response:
            return jsonify(auth)


if __name__ == '__main__':
    db.create_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
