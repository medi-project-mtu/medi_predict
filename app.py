from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

import pickle

std = pickle.load(open("std.pkl", "rb"))
present_model = pickle.load(open("svm_model.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Medi-Predict API</h1>"


@app.route("/predictDiabetes", methods=["POST"])
def predict():
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
