import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


df = pd.read_csv('diabetes.csv')

df.duplicated().sum()
df.drop_duplicates(inplace=True)


columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)

df.dropna(inplace=True)

X = df.drop('Outcome', axis=1)

scaler = StandardScaler()
# std = scaler.fit(X_train)

X = StandardScaler().fit_transform(X)


y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=[8]))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])


checkpointer = ModelCheckpoint(
    'diabetes.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=1, epochs=150,
                    validation_data=(X_test, y_test), callbacks=[checkpointer])
