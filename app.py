from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import pickle


app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Medi-Predict API</h1>"


@app.route("/predictAlzheimers", methods=["POST"])
def predictAlzheimers():
    ...


@app.route("/predictHeartDisease", methods=["POST"])
def predictHeartDisease():
    std_hd = pickle.load(open("models/heart_disease/std_heart.pkl", "rb"))
    present_model = pickle.load(
        open("models/heart_disease/SVC_heart_model_proba.pkl", "rb")
    )

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

    input_query = np.array(
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
    input_query = std_hd.transform(input_query)

    prediction = present_model.predict_proba(input_query)

    print(prediction[0][1])

    return jsonify({"Diagnosis": str(prediction[0][1])})


@app.route("/predictDiabetes", methods=["POST"])
def predictDiabetes():
    std_diabetes = pickle.load(open("models/diabletes/std.pkl", "rb"))
    present_model = pickle.load(open("models/diabetes/svm_model.pkl", "rb"))
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
    input_query = std_diabetes.transform(input_query)

    prediction = present_model.predict_proba(input_query)
    output = "{0:.{1}f}".format(prediction[0][1], 2)
    output_print = str(float(output) * 100) + "%"
    print(output_print)

    return jsonify({"Diagnosis": str(output_print)})


if __name__ == "__main__":
    app.run(debug=True)
