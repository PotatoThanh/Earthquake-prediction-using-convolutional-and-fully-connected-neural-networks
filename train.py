import os
from scipy.io import loadmat

import config
from data.earth_quake import get_data
from model.my_model import *

# Parameter
DATA = config.DATA
LEARNING_RATE = config.LEARNING_RATE
WINDOW_SIZE = config.WINDOW_SIZE

NUM_TRAIN = config.NUM_TRAIN
NUM_VAL = config.NUM_VAL
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
DELETE_TRAIN = config.DELETE_TRAIN
DELETE_TEST = config.DELETE_TEST


if DELETE_TRAIN:
    path_logs = './logs/*'
    path_ckpt = './checkpoint/*'
    print('rm -Rf' + path_logs)
    os.system('rm -Rf ' + path_logs)

    print('rm -Rf' + path_ckpt)
    os.system('rm -Rf ' + path_ckpt)

# load data
x_train, y_train, x_val, y_val, x_test, y_test, x_minmax, y_minmax = get_data()

x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# print # samples
print('Number train samples: ' + str(x_train.shape))
print('Number validation samples: ' + str(x_val.shape))
print('Number test samples: ' + str(x_test.shape))

# number inputs and outputs
FEATURE_SIZE = x_train.shape[1]
OUTPUT_SIZE = y_train.shape[1]
print('Number features: ' + str(FEATURE_SIZE))
print('Number outputs: ' + str(OUTPUT_SIZE))

# input shape
input_shape =(FEATURE_SIZE, 1)
number_output = OUTPUT_SIZE

# create model
model = CNN_model(input_shape, number_output)

# compile model with adam optimizer
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.mean_squared_error)

# call back functions
cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=20, batch_size=BATCH_SIZE,
                                             write_graph=True, write_grads=True,
                                             write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None)

cb_ckpt = keras.callbacks.ModelCheckpoint('./checkpoint/weights.h5', monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=False, mode='auto', period=1)
# train model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True,
          epochs=EPOCHS, validation_data=(x_val, y_val),
          callbacks=[cb_tensorboard, cb_ckpt])





