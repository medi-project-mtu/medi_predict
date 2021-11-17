import numpy as np
import pandas as pd
from sklearn import tree


def replace_zero(df):
    df_nan = df.copy(deep=True)
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_nan[cols] = df_nan[cols].replace({0: np.nan})
    return df_nan


def find_median(df, col):

    df_nondiab = df[df["Outcome"] == 0].reset_index(drop=True)
    df_diab = df[df["Outcome"] == 1].reset_index(drop=True)
    return (df_nondiab[col].median(), df_diab[col].median())


def replace_null(df, var):

    median_tuple = find_median(df, var)
    var_0 = median_tuple[0]
    var_1 = median_tuple[1]

    df.loc[(df["Outcome"] == 0) & (df[var].isnull()), var] = var_0
    df.loc[(df["Outcome"] == 1) & (df[var].isnull()), var] = var_1
    return df[var].isnull().sum()


def diabetesAI(input_query):
    df = pd.read_csv("testing/diabetes.csv")

    df_nan = replace_zero(df)

    replace_null(df_nan, "Glucose")
    replace_null(df_nan, "BloodPressure")
    replace_null(df_nan, "SkinThickness")
    replace_null(df_nan, "Insulin")
    replace_null(df_nan, "BMI")

    df = df_nan.copy()

    data = df.copy()

    Y = data["Outcome"]
    X = data.drop("Outcome", axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.20, random_state=20, stratify=Y
    # )

    clf = tree.DecisionTreeClassifier()

    clf.fit(X, Y)
    print(clf.score(X, Y))

    return clf.predict(input_query)
