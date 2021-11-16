from tensorflow import lite
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('diabetes.csv')

X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

# le = LabelEncoder()

# y = le.fit_transform(y)
# y = to_categorical(y)


model = Sequential()

model.add(Dense(8, activation='relu', input_shape=[8]))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['acc'])


model.fit(X, y, epochs=200)


# print("Accuracy of our model on test data : ",
#       model.evaluate(X, y)[1]*100, "%")

converter = lite.TFLiteConverter.from_keras_model(model)

tfmodel = converter.convert()

open('diabetes_.tflite', 'wb').write(tfmodel)
