
from tensorflow import lite
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

present_model = tf.keras.models.load_model('diabetes.h5')

# new_model = Sequential()
# new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
# new_model.add(Dense(1))
# copy weights


single_item_model = Sequential()
single_item_model.add(LSTM(20, batch_input_shape=(1, 1, 8), stateful=True))
single_item_model.add(Dense(8, activation='relu', input_shape=[8]))
single_item_model.add(Dense(8, activation='relu'))
single_item_model.add(Dense(4, activation='relu'))
single_item_model.add(Dense(1, activation='sigmoid'))

old_weights = present_model.get_weights()

single_item_model.set_weights(old_weights)
single_item_model.compile(loss='binary_crossentropy', optimizer='adam')

single_item_model.save('diabetes_1.h5')
