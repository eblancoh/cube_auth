# usr/bin/env python
import configparser
import itertools
import os
from urllib.parse import urlparse
import numpy as np
from pymongo import MongoClient
import sys
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support
import pickle
import json
from sklearn.metrics import confusion_matrix


def training_dataframe(mongodb_uri, split): 

    # Making a Connection with MongoClient
    client = MongoClient(mongodb_uri)
    o = urlparse(mongodb_uri).path[1::]
    # Getting a Database and parsing the name of the database from the MongoURI
    db = client[o]
    # Querying in MongoDB and obtaining the result as variable
    movements_collection = list(db.movements.find())

    movements = pd.DataFrame(movements_collection)

    gb_sequence = movements.groupby('sequence')
    # Split DataFrame grouped by sequences
    sequences = [gb_sequence.get_group(x) for x in gb_sequence.groups]

    info_bucket = list()
    # Recorremos cada uno de los dataframes en sequences
    for df in sequences:

        _id = pd.unique(df['_id'])[0]
        cube_type = pd.unique(df['cube_type'])[0]
        is_random = pd.unique(df['is_random'])[0]
        user = pd.unique(df['user_email'])[0]
        time = (df['timestamp'] - df['timestamp'].iloc[0]) / np.timedelta64(1, 's')

        num_moves = len(time)
        duration = time.iloc[-1]
        # Las diferencias de tiempo
        diff_time = np.diff(time)
        # Handling positioning
        yaw_pitch_roll = df['yaw_pitch_roll']
        if yaw_pitch_roll.iloc[0]:
            y = list(); p = list(); r = list()
            y_sigma = list(); p_sigma = list(); r_sigma = list()
            for bulk in yaw_pitch_roll:
                x = list(); y = list(); z = list()
                for item in bulk:
                    x.append(item['x']); y.append(item['y']); z.append(item['z'])
                    yaw = np.mean(x); pitch = np.mean(y); roll = np.mean(z)
                    yaw_std = np.std(x); pitch_std = np.std(y); roll_std = np.std(z);
                y.append(yaw); p.append(pitch); r.append(roll)
                y_sigma.append(yaw_std); p_sigma.append(pitch_std); r_sigma.append(roll_std)
        else:
            y = [0] * df.shape[0]
            p = [0] * df.shape[0]
            r = [0] * df.shape[0]
            y_sigma = [0] * df.shape[0]
            p_sigma = [0] * df.shape[0]
            r_sigma = [0] * df.shape[0]
        
        YAW = np.std(y); PITCH = np.std(p); ROLL = np.std(r)
        YAW_SIGMA = np.mean(y_sigma); PITCH_SIGMA = np.mean(p_sigma); ROLL_SIGMA = np.mean(r_sigma)
        
        try:
            # Nos interesan mínimo, máximo, Mediana, 25th percentile and 75th percentile
            q1 = np.percentile(diff_time, 25)
            q2 = np.percentile(diff_time, 50)
            q3 = np.percentile(diff_time, 75)
        except IndexError:
            q1 = q2 = q3 = 0
        iqr = (q3 - q1)/np.sqrt(num_moves)
        maximum = q3 + 1.58 * iqr
        minimum = q1 - 1.58 * iqr
        
        turn_values = ['11', '13', '21', '23', 
                       '31', '33', '41', '43', 
                       '51', '53', '61', '63']
        chk = df.groupby('turn').size()
        chk_index = chk.index

        for i in turn_values:
            if i not in chk_index:
                chk.loc[i] = 0
        
        no_11 = chk.loc['11']
        no_13 = chk.loc['13']
        no_21 = chk.loc['21']
        no_23 = chk.loc['23']
        no_31 = chk.loc['31']
        no_33 = chk.loc['33']
        no_41 = chk.loc['41']
        no_43 = chk.loc['43']
        no_51 = chk.loc['51']
        no_53 = chk.loc['53']
        no_61 = chk.loc['61']
        no_63 = chk.loc['63']

        no_CW = no_11 + no_21 + no_31 + no_41 + no_51 + no_61
        no_CCW = no_13 + no_23 + no_33 + no_43 + no_53 + no_63

        # Definimos un diccionario vacío
        df_dict = {}

        df_dict['user_email'] = user
        df_dict['cube_type'] = cube_type
        df_dict['is_random'] = is_random
        df_dict['num_moves'] = num_moves
        df_dict['duration'] = duration
        df_dict['q1'] = q1
        df_dict['q2'] = q2
        df_dict['q3'] = q3
        df_dict['iqr'] = iqr
        df_dict['maximum'] = maximum
        df_dict['minimum'] = minimum
        df_dict['no_11'] = no_11 / num_moves
        df_dict['no_13'] = no_13 / num_moves
        df_dict['no_21'] = no_21 / num_moves
        df_dict['no_23'] = no_23 / num_moves
        df_dict['no_31'] = no_31 / num_moves
        df_dict['no_33'] = no_33 / num_moves
        df_dict['no_41'] = no_41 / num_moves
        df_dict['no_43'] = no_43 / num_moves
        df_dict['no_51'] = no_51 / num_moves
        df_dict['no_53'] = no_53 / num_moves
        df_dict['no_61'] = no_61 / num_moves
        df_dict['no_63'] = no_63 / num_moves
        df_dict['no_CW'] = no_CW / num_moves
        df_dict['no_CCW'] = no_CCW / num_moves
        # Handling positioning
        df_dict['yaw'] = YAW
        df_dict['pitch'] = PITCH
        df_dict['roll'] = ROLL
        df_dict['yaw-sigma'] = YAW_SIGMA
        df_dict['pitch-sigma'] = PITCH_SIGMA
        df_dict['roll-sigma'] = ROLL_SIGMA

        # Partimos cada dataset df en sequences en n_split parts
        n_split = split
        df_splitted = np.array_split(df, n_split)

        i=0
        if n_split == 1:
            info_bucket.append(df_dict)
            continue
        else:
            for elem in df_splitted:
                if len(elem) > 1:

                    _time = (elem['timestamp'] - elem['timestamp'].iloc[0]) / np.timedelta64(1, 's')
                    _num_moves = len(_time)
                    _duration = _time.iloc[-1]
                    # Las diferencias de tiempo se calculan aquí
                    _diff_time = np.diff(_time)

                    try:
                        # Nos interesan mínimo, máximo, Mediana, 25th percentile and 75th percentile
                        _q1 = np.percentile(_diff_time, 25)
                        _q2 = np.percentile(_diff_time, 50)
                        _q3 = np.percentile(_diff_time, 75)
                    except IndexError:
                        _q1 = _q2 = _q3 = 0
                    _iqr = (_q3 - _q1)/np.sqrt(_num_moves)
                    _minimum = _q1 - 1.58 * _iqr
                    _maximum = _q3 + 1.58 * _iqr

                    _turn_values = ['11', '13', '21', '23', 
                                    '31', '33', '41', '43', 
                                    '51', '53', '61', '63']
                    _chk = elem.groupby('turn').size()
                    _chk_index = _chk.index

                    for k in _turn_values:
                        if k not in _chk_index:
                            _chk.loc[k] = 0
                    
                    _no_11 = _chk.loc['11']
                    _no_13 = _chk.loc['13']
                    _no_21 = _chk.loc['21']
                    _no_23 = _chk.loc['23']
                    _no_31 = _chk.loc['31']
                    _no_33 = _chk.loc['33']
                    _no_41 = _chk.loc['41']
                    _no_43 = _chk.loc['43']
                    _no_51 = _chk.loc['51']
                    _no_53 = _chk.loc['53']
                    _no_61 = _chk.loc['61']
                    _no_63 = _chk.loc['63']

                    _no_CW = _no_11 + _no_21 + _no_31 + _no_41 + _no_51 + _no_61
                    _no_CCW = _no_13 + no_23 + _no_33 + _no_43 + _no_53 + _no_63

                    df_dict['num.moves.{0}'.format(i)] = _num_moves
                    df_dict['duration.{0}'.format(i)] = _duration
                    df_dict['q1.{0}'.format(i)] = _q1
                    df_dict['q2.{0}'.format(i)] = _q2
                    df_dict['q3.{0}'.format(i)] = _q3
                    df_dict['iqr.{0}'.format(i)] = _iqr
                    df_dict['maximum.{0}'.format(i)] = _maximum
                    df_dict['minimum.{0}'.format(i)] = _minimum
                    df_dict['no_11.{0}'.format(i)] = _no_11 / _num_moves
                    df_dict['no_13.{0}'.format(i)] = _no_13 / _num_moves
                    df_dict['no_21.{0}'.format(i)] = _no_21 / _num_moves
                    df_dict['no_23.{0}'.format(i)] = _no_23 / _num_moves
                    df_dict['no_31.{0}'.format(i)] = _no_31 / _num_moves
                    df_dict['no_33.{0}'.format(i)] = _no_33 / _num_moves
                    df_dict['no_41.{0}'.format(i)] = _no_41 / _num_moves
                    df_dict['no_43.{0}'.format(i)] = _no_43 / _num_moves
                    df_dict['no_51.{0}'.format(i)] = _no_51 / _num_moves
                    df_dict['no_53.{0}'.format(i)] = _no_53 / _num_moves
                    df_dict['no_61.{0}'.format(i)] = _no_61 / _num_moves
                    df_dict['no_63.{0}'.format(i)] = _no_63 / _num_moves
                    df_dict['no_CW.{0}'.format(i)] = _no_CW / _num_moves
                    df_dict['no_CCW.{0}'.format(i)] = _no_CCW / _num_moves

                    i+=1
                
                else:
                    continue

            
            info_bucket.append(df_dict)

    dataframe = pd.DataFrame(info_bucket)
    dataframe = dataframe.fillna(value=0)
    return dataframe


def testing_dataframe(body, split):

    user = json.loads(body.decode('utf-8'))['email']
    login_id = json.loads(body.decode('utf-8'))['id']
    movements = json.loads(body.decode('utf-8'))['movements']

    cube_type = movements[0]['cube_type']
    is_random = movements[0]['is_random']

    # Pasamos todos los movimientos a un pandas DataFrame

    movements = pd.DataFrame(movements)
    movements = movements.convert_objects(convert_numeric=True)
    movements['timestamp'] = pd.to_datetime(movements['timestamp'], infer_datetime_format=True)
    time = (movements['timestamp'] - movements['timestamp'].iloc[0]) / np.timedelta64(1, 's')
    num_moves = len(time)
    duration = time.iloc[-1]
    # Las diferencias de tiempo
    diff_time = np.diff(time)

    # Handling positioning
    yaw_pitch_roll = movements['yaw_pitch_roll']
    if yaw_pitch_roll.iloc[0]:
        y = list(); p = list(); r = list()
        y_sigma = list(); p_sigma = list(); r_sigma = list()
        for bulk in yaw_pitch_roll:
            x = list(); y = list(); z = list()
            for item in bulk:
                x.append(item['x']); y.append(item['y']); z.append(item['z'])
                yaw = np.mean(x); pitch = np.mean(y); roll = np.mean(z)
                yaw_std = np.std(x); pitch_std = np.std(y); roll_std = np.std(z);
            y.append(yaw); p.append(pitch); r.append(roll)
            y_sigma.append(yaw_std); p_sigma.append(pitch_std); r_sigma.append(roll_std)
    else:
        y = [0] * movements.shape[0]
        p = [0] * movements.shape[0]
        r = [0] * movements.shape[0]
        y_sigma = [0] * movements.shape[0]
        p_sigma = [0] * movements.shape[0]
        r_sigma = [0] * movements.shape[0]
    
    YAW = np.std(y); PITCH = np.std(p); ROLL = np.std(r)
    YAW_SIGMA = np.mean(y_sigma); PITCH_SIGMA = np.mean(p_sigma); ROLL_SIGMA = np.mean(r_sigma)

    
    info_bucket = list()

    try:
        # Nos interesan mínimo, máximo, Mediana, 25th percentile and 75th percentile
        q1 = np.percentile(diff_time, 25)
        q2 = np.percentile(diff_time, 50)
        q3 = np.percentile(diff_time, 75)
    except IndexError:
        q1 = q2 = q3 = 0
    iqr = (q3 - q1)/np.sqrt(num_moves)
    maximum = q3 + 1.58 * iqr
    minimum = q1 - 1.58 * iqr
    
    turn_values = ['11', '13', '21', '23', 
                '31', '33', '41', '43', 
                '51', '53', '61', '63']
    chk = movements.groupby('turn').size()
    chk_index = chk.index

    for i in turn_values:
        if i not in chk_index:
            chk.loc[i] = 0
    
    no_11 = chk.loc['11']
    no_13 = chk.loc['13']
    no_21 = chk.loc['21']
    no_23 = chk.loc['23']
    no_31 = chk.loc['31']
    no_33 = chk.loc['33']
    no_41 = chk.loc['41']
    no_43 = chk.loc['43']
    no_51 = chk.loc['51']
    no_53 = chk.loc['53']
    no_61 = chk.loc['61']
    no_63 = chk.loc['63']

    no_CW = no_11 + no_21 + no_31 + no_41 + no_51 + no_61
    no_CCW = no_13 + no_23 + no_33 + no_43 + no_53 + no_63

    # Definimos un diccionario vacío
    df_dict = {}

    df_dict['user_email'] = user
    df_dict['cube_type'] = cube_type
    df_dict['is_random'] = is_random
    df_dict['num_moves'] = num_moves
    df_dict['duration'] = duration
    df_dict['q1'] = q1
    df_dict['q2'] = q2
    df_dict['q3'] = q3
    df_dict['iqr'] = iqr
    df_dict['maximum'] = maximum
    df_dict['minimum'] = minimum
    df_dict['no_11'] = no_11 / num_moves
    df_dict['no_13'] = no_13 / num_moves
    df_dict['no_21'] = no_21 / num_moves
    df_dict['no_23'] = no_23 / num_moves
    df_dict['no_31'] = no_31 / num_moves
    df_dict['no_33'] = no_33 / num_moves
    df_dict['no_41'] = no_41 / num_moves
    df_dict['no_43'] = no_43 / num_moves
    df_dict['no_51'] = no_51 / num_moves
    df_dict['no_53'] = no_53 / num_moves
    df_dict['no_61'] = no_61 / num_moves
    df_dict['no_63'] = no_63 / num_moves
    df_dict['no_CW'] = no_CW / num_moves
    df_dict['no_CCW'] = no_CCW / num_moves
    # Handling positioning
    df_dict['yaw'] = YAW
    df_dict['pitch'] = PITCH
    df_dict['roll'] = ROLL
    df_dict['yaw-sigma'] = YAW_SIGMA
    df_dict['pitch-sigma'] = PITCH_SIGMA
    df_dict['roll-sigma'] = ROLL_SIGMA

    # Partimos cada dataset df en sequences en n_split parts
    n_split = split
    df_splitted = np.array_split(movements, n_split)

    i=0
    if n_split == 1:
        info_bucket.append(df_dict)
    else:
        for elem in df_splitted:
            if len(elem) > 1:

                _time = (elem['timestamp'] - elem['timestamp'].iloc[0]) / np.timedelta64(1, 's')
                _num_moves = len(_time)
                _duration = _time.iloc[-1]
                # Las diferencias de tiempo
                _diff_time = np.diff(_time)

                try:
                    # Nos interesan mínimo, máximo, Mediana, 25th percentile and 75th percentile
                    _q1 = np.percentile(_diff_time, 25)
                    _q2 = np.percentile(_diff_time, 50)
                    _q3 = np.percentile(_diff_time, 75)
                except IndexError:
                    _q1 = _q2 = _q3 = 0
                _iqr = (_q3 - _q1)/np.sqrt(_num_moves)
                _minimum = _q1 - 1.58 * _iqr
                _maximum = _q3 + 1.58 * _iqr

                _turn_values = ['11', '13', '21', '23', 
                                '31', '33', '41', '43', 
                                '51', '53', '61', '63']
                _chk = elem.groupby('turn').size()
                _chk_index = _chk.index

                for k in _turn_values:
                    if k not in _chk_index:
                        _chk.loc[k] = 0
                
                _no_11 = _chk.loc['11']
                _no_13 = _chk.loc['13']
                _no_21 = _chk.loc['21']
                _no_23 = _chk.loc['23']
                _no_31 = _chk.loc['31']
                _no_33 = _chk.loc['33']
                _no_41 = _chk.loc['41']
                _no_43 = _chk.loc['43']
                _no_51 = _chk.loc['51']
                _no_53 = _chk.loc['53']
                _no_61 = _chk.loc['61']
                _no_63 = _chk.loc['63']

                _no_CW = _no_11 + _no_21 + _no_31 + _no_41 + _no_51 + _no_61
                _no_CCW = _no_13 + no_23 + _no_33 + _no_43 + _no_53 + _no_63

                df_dict['num.moves.{0}'.format(i)] = _num_moves
                df_dict['duration.{0}'.format(i)] = _duration
                df_dict['q1.{0}'.format(i)] = _q1
                df_dict['q2.{0}'.format(i)] = _q2
                df_dict['q3.{0}'.format(i)] = _q3
                df_dict['iqr.{0}'.format(i)] = _iqr
                df_dict['maximum.{0}'.format(i)] = _maximum
                df_dict['minimum.{0}'.format(i)] = _minimum
                df_dict['no_11.{0}'.format(i)] = _no_11 / _num_moves
                df_dict['no_13.{0}'.format(i)] = _no_13 / _num_moves
                df_dict['no_21.{0}'.format(i)] = _no_21 / _num_moves
                df_dict['no_23.{0}'.format(i)] = _no_23 / _num_moves
                df_dict['no_31.{0}'.format(i)] = _no_31 / _num_moves
                df_dict['no_33.{0}'.format(i)] = _no_33 / _num_moves
                df_dict['no_41.{0}'.format(i)] = _no_41 / _num_moves
                df_dict['no_43.{0}'.format(i)] = _no_43 / _num_moves
                df_dict['no_51.{0}'.format(i)] = _no_51 / _num_moves
                df_dict['no_53.{0}'.format(i)] = _no_53 / _num_moves
                df_dict['no_61.{0}'.format(i)] = _no_61 / _num_moves
                df_dict['no_63.{0}'.format(i)] = _no_63 / _num_moves
                df_dict['no_CW.{0}'.format(i)] = _no_CW / _num_moves
                df_dict['no_CCW.{0}'.format(i)] = _no_CCW / _num_moves

                i+=1
            
            else:
                continue

        
        info_bucket.append(df_dict)
    dataframe = pd.DataFrame(info_bucket)
    dataframe = dataframe.fillna(value=0)
    dataframe = dataframe.drop(['user_email', 'cube_type', 'is_random'], axis=1)
    return user, dataframe, login_id


def user_to_binary(dataframe, user):
    transform_user = dataframe['user_email']==user
    dataframe['user'] = transform_user.astype('int')
    return dataframe


def obtain_features(dataframe, upsampling=True, test_size=0.2):

    if upsampling:
        dataframe = upsample(dataframe)
    else:
        dataframe = downsample(dataframe)
    
    X = dataframe.drop(['user', 'user_email', 'cube_type', 'is_random'], axis=1)
    Y = dataframe['user']
    
    # Hacemos la partición de entrenamiento y validación
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)

    return X_train, X_test, Y_train, Y_test


def upsample(df):
    from sklearn.utils import resample

    # Separate majority and minority classes
    df_majority = df[df.user==0]
    df_minority = df[df.user==1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=df_majority.__len__(),    # to match majority class
                                    random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled


def downsample(df):
    from sklearn.utils import resample

    # Separate majority and minority classes
    df_majority = df[df.user==0]
    df_minority = df[df.user==1]
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=df_minority.__len__(),     # to match minority class
                                    random_state=123) # reproducible results
    
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled


def ml_engine_training(x_train, x_test, y_train, y_test, user, model):
    if model == 'logRegr':
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, class_weight=None, solver='newton-cg', max_iter=100, penalty='l2')
        model.fit(x_train, y_train)

        model.score(x_test, y_test)
        
        y_pred = model.predict(x_test)

        filename = 'logRegr_' + user +'.sav'

    elif model == 'SVC':
        from sklearn.svm import SVC

        model = SVC(C=1.0, gamma=0.1, kernel='rbf', probability=True)
        model.fit(x_train, y_train)

        model.score(x_test, y_test)
        
        y_pred = model.predict(x_test)

        filename = 'SVC_' + user +'.sav'
    
    y_true = np.array(y_test)
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[1, 0])

    # Confusion matrix
    _conf = confusion_matrix(y_true, y_pred).ravel()

    # Contamos el número real de positivos y negativos
    pos = np.count_nonzero(y_true)
    neg = y_true.__len__() - pos

    # Guardamos checkpoint
    
    pickle.dump(model, open(filename, 'wb'))

    print('Precision: ', metrics[0][0], ' ||', 'Recall: ', metrics[1][0], ' ||', 'F-Score: ', metrics[2][0])
    print('tn: ', _conf[0], ' || fp: ', _conf[1], ' || fn: ', _conf[2],  ' || tp: ', _conf[3], ' || pos: ', pos, ' || neg:', neg)
    return metrics[0][0], metrics[1][0], metrics[2][0]


def model_testing(dataframe, user, model):

    if model == 'logRegr':
        filename = 'logRegr_' + user + '.sav'
    elif model == 'SVC':
        filename = 'SVC_' + user + '.sav'
    
    if not filename in os.listdir(os.getcwd()):
        sys.exit("No training has been launched before or this user does not exist in database")
    else:
        loaded_model = pickle.load(open(filename, 'rb'))

        # Cargamos el escalado desde el fichero scaler.sav y escalamos la secuencia de testeo
        dataframe = load_scaling(dataframe)

        probability = loaded_model.predict_proba(dataframe)[0][1]
        # Devolvemos la probabilidad de que sea un 1
        return probability


def save_scaling(dataframe, scalerfile='scaler.sav'):
    from sklearn.preprocessing import StandardScaler

    # The StandardScaler model is stored for further testing
    scaler = StandardScaler()

    try:
        to_scale = dataframe[dataframe.columns.difference(['user', 'user_email', 'cube_type', 'is_random'])]
        misc = dataframe[['user', 'user_email', 'cube_type', 'is_random']]

        scaling = scaler.fit(to_scale.values)

        pickle.dump(scaling, open(scalerfile, 'wb'))
        # Transformamos el dataframe entre -1 y 1
        dataframe = pd.DataFrame(scaler.fit_transform(to_scale), columns=to_scale.keys())
        # Concatenamos y ya lo tenemos normalizado
        dataframe = pd.concat([dataframe, misc], axis=1)

    except KeyError:
        scaling = scaler.fit(dataframe.values)

        pickle.dump(scaling, open(scalerfile, 'wb'))
        # Transformamos el dataframe entre -1 y 1
        dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.keys())

    return dataframe


def load_scaling(dataframe, scalerfile='scaler.sav'):

    scaler = pickle.load(open(scalerfile, 'rb'))

    # Hacemos una transformación de los datos en base al escalado ya guardado
    dataframe = pd.DataFrame(scaler.transform(dataframe))

    print(dataframe)

    return dataframe


