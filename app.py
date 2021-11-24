from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import pickle

std = pickle.load(open("std.pkl", "rb"))
present_model = pickle.load(open("svm_model.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Medi-Predict API</h1>"


@app.route("/predictAlzheimers", methods=["POST"])
def predictAlzheimers():
    ...


@app.route("/predictHeartDisease", methods=["POST"])
def predictHeartDisease():
    df = pd.read_csv("testing/cleveland.csv")
    df = df.dropna()
    Y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    age = request.form.get("age")
    sex = request.form.get("sex")
    chest_pain_type = request.form.get("chestPainType")
    rbp = request.form.get("rbp")
    serum_chol = request.form.get("serumChol")
    fbs = request.form.get("fbs")
    resting_ecg = request.form.get("restingECG")
    max_heart_rate = request.form.get("maxHeartRate")
    exercise_induced_angina = request.form.get("exerciseInducedAngina")
    st_depression = request.form.get("stDepression")
    peak_exercise_st_segment = request.form.get("peakExerciseSTSegment")
    major_vessels = request.form.get("majorVessels")
    thal = request.form.get("thal")

    pred = clf.predict(
        [
            [
                age,
                sex,
                chest_pain_type,
                rbp,
                serum_chol,
                fbs,
                resting_ecg,
                max_heart_rate,
                exercise_induced_angina,
                st_depression,
                peak_exercise_st_segment,
                major_vessels,
                thal,
            ]
        ]
    )
    print(pred)
    return jsonify({"Diagnosis": str(pred[0])})


@app.route("/predictDiabetes", methods=["POST"])
def predictDiabetes():
    pregnancies = request.form.get("pregnancies")
    glucose = request.form.get("glucose")
    bp = request.form.get("bp")
    skinThickness = request.form.get("skinThickness")
    insulin = request.form.get("insulin")
    bmi = request.form.get("bmi")
    dpf = request.form.get("dpf")
    age = request.form.get("age")

    input_query = np.array(
        [[pregnancies, glucose, bp, skinThickness, insulin, bmi, dpf, age]]
    )
    input_query = std.transform(input_query)

    prediction = present_model.predict_proba(input_query)
    output = "{0:.{1}f}".format(prediction[0][1], 2)
    output_print = str(float(output) * 100) + "%"
    print(output_print)

    return jsonify({"Diagnosis": str(output_print)})


if __name__ == "__main__":
    app.run(debug=True)
