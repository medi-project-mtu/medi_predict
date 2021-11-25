import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
# from pywt import wavedec
from functools import reduce
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras as K
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_validate
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Conv1D,Conv2D,Add
from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
from sklearn.impute  import SimpleImputer
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
import pickle

data_cross = pd.read_csv("model notebooks/alzheimer/oasis_cross-sectional.csv")
data_long = pd.read_csv("model notebooks/alzheimer/oasis_longitudinal.csv")

data_cross.dropna(subset=['CDR'],inplace=True)

data_cross.drop(columns=['ID','Delay'],inplace=True)
data_long = data_long.rename(columns={'EDUC':'Educ'})
data_long.drop(columns=['Subject ID','MRI ID','Group','Visit','MR Delay'],inplace=True)

data = pd.concat([data_cross,data_long])

# cor = data.corr()
# plt.figure(figsize=(12,9))
# sns.heatmap(cor, xticklabels=cor.columns.values,yticklabels=cor.columns.values, annot=True)

imputer = SimpleImputer ( missing_values = np.nan,strategy='most_frequent')

imputer.fit(data[['SES']])
data[['SES']] = imputer.fit_transform(data[['SES']])

# We perform it with the median
imputer = SimpleImputer ( missing_values = np.nan,strategy='median')

imputer.fit(data[['MMSE']])
data[['MMSE']] = imputer.fit_transform(data[['MMSE']])

le = preprocessing.LabelEncoder()
data['CDR'] = le.fit_transform(data['CDR'].values)

data['M/F'] = data['M/F'].map({'M':0,'F':1})
data.pop('Hand')

data = pd.get_dummies(data)

data = data.drop(data[data['CDR']==3].index)

y = data.pop('CDR')
x = data
columns = x.columns

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 42,test_size=0.3)

scaler = StandardScaler()

std = scaler.fit(X_train)

X_train = std.transform(X_train)
X_test = std.transform(X_test)
X_train = pd.DataFrame(X_train, columns = columns)
X_test = pd.DataFrame(X_test, columns = columns)


pickle.dump(std, open('models/alzheimer/std_alz_cleaned.pkl','wb'))

FOLDS =10

parametros_gb = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.005,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],
    "min_samples_split": [0.01, 0.025, 0.005,0.4,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],
    "min_samples_leaf": [1,2,3,5,8,10,15,20,40,50,55,60,65,70,80,85,90,100],
    "max_depth":[3,5,8,10,15,20,25,30,40,50],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":range(1,100)
    }

model_gb = GradientBoostingClassifier()


gb_random = RandomizedSearchCV(estimator = model_gb, param_distributions = parametros_gb, n_iter = 100, cv = FOLDS, 
                               verbose=0, random_state=42,n_jobs = -1, scoring='accuracy')
gb_random.fit(X_train, y_train)

gb_random.best_params_

model_gb = gb_random.best_estimator_

pickle.dump(model_gb, open('models/alzheimer/gb_model_cleaned.pkl','wb'))

print(model_gb.score(X_test,y_test))

cross_val_score(model_gb, x, y, cv=10, scoring='accuracy').mean()