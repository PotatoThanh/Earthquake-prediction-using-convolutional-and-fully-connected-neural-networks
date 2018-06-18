import config
import numpy as np
from scipy.io import loadmat

DATA = config.DATA
LEARNING_RATE = config.LEARNING_RATE
WINDOW_SIZE = config.WINDOW_SIZE

NUM_TRAIN = config.NUM_TRAIN
NUM_VAL = config.NUM_VAL
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
DELETE_TRAIN = config.DELETE_TRAIN
DELETE_TEST = config.DELETE_TEST

# Data names
TEST_X = config.TEST_X
TEST_X_1STATN = config.TEST_X_1STATN
TEST_X_3STATNs = config.TEST_X_3STATNs
TEST_Y_ADO = config.TEST_Y_ADO
TEST_Y_DEPTH = config.TEST_Y_DEPTH
TEST_Y_EQLOC = config.TEST_Y_EQLOC
TEST_Y_MAG = config.TEST_Y_MAG
TEST_Y_RPV = config.TEST_Y_RPV
TEST_Y_RSS = config.TEST_Y_RSS
TEST_Y_USC = config.TEST_Y_USC

def get_minmax(data):
    minmax = list()
    for i in range(len(data[0])):
        col_values = [row[i] for row in data]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def get_normalize(data, minmax):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return data

def get_denormalize(data, minmax):
    for row in data:
        for i in range(len(row)):
            row[i] = row[i] * (minmax[i][1] - minmax[i][0]) + minmax[i][0]
    return data

def normalize(data):
    minmax = get_minmax(data)
    data_norm = get_normalize(data, minmax)

    return data_norm, minmax

def denormalize(data_norm, minmax):
    data = get_denormalize(data_norm, minmax)

    return data

def load_data():
    # Data names
    x = loadmat(TEST_X)
    x_1statn = loadmat(TEST_X_1STATN)
    x_3statns = loadmat(TEST_X_3STATNs)

    y_ADO = loadmat(TEST_Y_ADO)
    y_depth = loadmat(TEST_Y_DEPTH)
    y_epLoc = loadmat(TEST_Y_EQLOC)
    y_mag = loadmat(TEST_Y_MAG)
    y_RPV = loadmat(TEST_Y_RPV)
    y_RSS = loadmat(TEST_Y_RSS)
    y_USC = loadmat(TEST_Y_USC)

    x_data = np.concatenate((x['data'], x_1statn['data_1statn'], x_3statns['data_3statns']), axis=1)
    y_data = np.concatenate((y_ADO['sWave_ADO'], y_depth['data_depth'], y_epLoc['data_eqLoc'],
                             y_mag['data_mag'], y_RPV['sWave_RPV'], y_RSS['sWave_RSS'], y_USC['sWave_USC']), axis=1)

    return (x_data, y_data)

def get_data():
    x_data, y_data = load_data()

    # x_data, x_minmax = normalize(x_data)
    # y_data, y_minmax = normalize(y_data)

    x_minmax = 0
    y_minmax = 0

    x_train = x_data[0:NUM_TRAIN, :]
    y_train = y_data[0:NUM_TRAIN, :]

    x_val = x_data[NUM_TRAIN:NUM_TRAIN+NUM_VAL, :]
    y_val = y_data[NUM_TRAIN:NUM_TRAIN+NUM_VAL, :]

    x_test = x_data[NUM_TRAIN + NUM_VAL:, :]
    y_test = y_data[NUM_TRAIN + NUM_VAL:, :]

    return (x_train, y_train, x_val, y_val, x_test, y_test, x_minmax, y_minmax)

