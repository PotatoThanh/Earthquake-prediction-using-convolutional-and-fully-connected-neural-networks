import os
from scipy.io import loadmat
from sklearn import preprocessing

import config
from data.earth_quake import *
from model.my_model import *


x_data, y_data = load_data()
minmax1 = get_minmax(y_data)
norm_data = get_normalize(y_data, minmax1)
print(minmax1)


minmax = get_minmax(norm_data)
print(minmax)

denorm = get_denormalize(norm_data, minmax1)
minmax = get_minmax(denorm)
print(minmax)