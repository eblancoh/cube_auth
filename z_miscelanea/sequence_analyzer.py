from pandas import DataFrame
from pandas.io.json import json_normalize
import json
import pandas as pd
import numpy as np
import os


def json_parser(json_file):
    """
    For training
    :param json_file:
    :return:
    """

    # Read the .json file to format it correctly
    with open(json_file, 'r') as item:
        file = item.read()
    item.close()

    if '\n' in file and file[0] == '{':
        # Replace '\n' with commas
        file = file.replace('\n', ',')
        # Open and close the string with brackets
        file = '[' + file[0:-1] + ']'
    elif '}},{' in file and file[0] == '{':
        file = '[' + file + ']'

    # Load the string as a .json file element and normalize for every nested element
    df = json_normalize(json.loads(file))
    return df


def feat_df_train(df):
    """
    Single feature extraction
    :param df:
    :param pad_length:
    :return:
    """

    # turn_ok = df.loc[True == df['data.sync'], :]

    # Following data can be stored after turn_ok is identified:
    # Timestamps of every valid movement
    # Sequence of shifts
    # time_stamp = turn_ok['stamp']
    # shift = turn_ok['data.code']
    time_stamp = df['stamp']
    shift = df['data.code']
    faces = df['data.side']
    sentido = df['data.shift']

    # To avoid time_stamp handling, format is specified to Pandas
    time_stamp = pd.to_datetime(time_stamp, format='%Y-%m-%dT%H:%M:%S.%fZ')
    time_stamp = (time_stamp - time_stamp[0]) / np.timedelta64(1, 's')

    # Possible values considered for shift information provided in
    dict_mapping = {'0xA0':  1, '0xA1':  2, '0xA2':  3, '0xA3':  4,
                    '0xB0':  5, '0xB1':  6, '0xB2':  7, '0xB3':  8,
                    '0xC0':  9, '0xC1': 10, '0xC2': 11, '0xC3': 12,
                    '0xD0': 13, '0xD1': 14, '0xD2': 15, '0xD3': 16,
                    '0xE0': 17, '0xE1': 18, '0xE2': 19, '0xE3': 20,
                    '0xF0': 21, '0xF1': 22, '0xF2': 23, '0xF3': 24}

    # The shift variable is mapped according to described criteria
    shift = shift.map(dict_mapping)

    dict_mapping =  {'CW': 0, '2CW': 1, 'CCW': 2, '2CCW': 3}

    sentido = sentido.map(dict_mapping)

    return time_stamp, shift, faces, sentido

times = []
codes = []
faces = []
shifts = []
for filename in os.listdir('/home/eblancoh/Desktop/json_examples/Leire/'):

    df = json_parser(os.path.join('/home/eblancoh/Desktop/json_examples/Leire', filename))

    df = df[pd.notnull(df['data.code'])]

    time_stamp, shift, face, sentido = feat_df_train(df=df)

    times.append(time_stamp)
    codes.append(shift)
    faces.append(face)
    shifts.append(sentido)

# Algunas estadísticas curiosas
[print(len(i)) for i in times]
[print(max(i)) for i in times]
[print(len(i)/max(i)) for i in times]

import matplotlib.pyplot as plt

# Calculate number of bins based on binsize for both x and y
min_x_data, max_x_data = -1, 6
binsize = 1.0
num_x_bins = np.floor((max_x_data - min_x_data) / binsize)

min_y_data, max_y_data = -1, 4
binsize = 1.0
num_y_bins = np.floor((max_y_data - min_y_data) / binsize)

# Axes definitions
nullfmt = plt.NullFormatter()
left, width = 0.1, 0.4
bottom, height = 0.1, 0.4
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.4]
rect_histy = [left_h, bottom, 0.4, height]

# Generate initial figure, scatter plot, and histogram quadrants
fig = plt.figure(221)

axScatter = fig.add_subplot(223, position=rect_scatter)
axScatter.set_xlabel('x')
axScatter.set_ylabel('y')
axScatter.set_xlim(-1, 6)
axScatter.set_ylim(-1, 4)

axHistX = fig.add_subplot(221, position=rect_histx)
axHistX.set_xlim(-1, 6)

axHistY = fig.add_subplot(224, position=rect_histy)
axHistY.set_ylim(-1, 4)

# Remove labels from histogram edges touching scatter plot
axHistX.xaxis.set_major_formatter(nullfmt)
axHistY.yaxis.set_major_formatter(nullfmt)

# Draw scatter plot
axScatter.scatter(faces[0], shifts[0], marker='o', color = 'darkblue', edgecolor='none', s=5, alpha=1)

# Draw x-axis histogram
axHistX.hist(faces[0], int(num_x_bins), ec='green', fc='none', histtype='bar', align = 'left')


# Draw y-axis histogram
axHistY.hist(shifts[0], int(num_y_bins), ec='green', fc='none', histtype='bar', orientation='horizontal', align = 'left')




for i in range(len(times)):
    plt.scatter(faces[i], shifts[i], s=1)

plt.show()



