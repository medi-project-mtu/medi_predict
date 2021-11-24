import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn import svm
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("testing/cleveland.csv")


# we can dropna to the entire df cus there are only 6 null values in 2 columns, so tiny it's negligible
df = df.dropna()


# def replace_zero(df):
#     df_nan = df.copy(deep=True)
#     cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
#     df_nan[cols] = df_nan[cols].replace({0: np.nan})
#     return df_nan


# df_nan = replace_zero(df)


# def find_median(df, col):

#     df_nondiab = df[df["Outcome"] == 0].reset_index(drop=True)
#     df_diab = df[df["Outcome"] == 1].reset_index(drop=True)
#     return (df_nondiab[col].median(), df_diab[col].median())


# def replace_null(df, var):

#     median_tuple = find_median(df, var)
#     var_0 = median_tuple[0]
#     var_1 = median_tuple[1]

#     df.loc[(df["Outcome"] == 0) & (df[var].isnull()), var] = var_0
#     df.loc[(df["Outcome"] == 1) & (df[var].isnull()), var] = var_1

#     return df[var].isnull().sum()


# replace_null(df_nan, "Glucose")
# replace_null(df_nan, "BloodPressure")
# replace_null(df_nan, "SkinThickness")
# replace_null(df_nan, "Insulin")
# replace_null(df_nan, "BMI")

# df = df_nan.copy()

# data = df.copy()

Y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# columns = X.columns
# print(columns)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.20, random_state=20, stratify=Y
# )

clf = DecisionTreeClassifier()

clf.fit(X, Y)

pred = clf.predict([[67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3]])
print(pred)
# clf.fit(X_train, y_train)

# print(clf.predict([[1, 122, 90, 51, 220, 49.7, 0.325, 31]]))
# =================================

# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# X_train = clf.transform(X_train)


# scaler = StandardScaler()
# std = scaler.fit(X_train)

# joblib.dump(std, open("stdHeartDisease.pkl", "wb"))

# X_train = std.transform(X_train)
# X_test = std.transform(X_test)
# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)

# svm_model = svm.SVC(probability=True).fit(X_train, y_train)
# svm_pred = svm_model.predict(X_test)
# svm_model.score(X_test, y_test)

# pickle.dump(clf, open("heartDiseaseDT.pkl", "wb"))


# def build_model():
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(8, activation='relu', input_shape=[
#                           len(X_train.keys())]),
#     #tf.keras.layers.Dense(4, activation='relu'),
#     tf.keras.layers.Dense(4, activation='relu'),
#     tf.keras.layers.Dense(2, activation='relu'),

#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizer, metrics=['accuracy'])
# return model


# neural_model = build_model()

# checkpoint_name = 'Weights_raw\Weights_raw-{epoch:03d}--{val_accuracy:.5f}.hdf5'

# # checkpoint_name = "Weight\diabetes_raw.h5"
# checkpoint = ModelCheckpoint(
#     checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# callbacks_list = [checkpoint]

# EPOCHS = 1000
# neural_pred = neural_model.fit(X_train, y_train, epochs=EPOCHS,
#                                validation_split=0.15, verbose=2, callbacks=callbacks_list)
