import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "0.2% Proof Stress (MPa)  =  {}".format(prediction[0][0]),
                           prediction_text1 = "Tensile Strength (MPa)  =  {}".format(prediction[0][1]),
                           prediction_text2 = " Elongation (%)  =  {}".format(prediction[0][2]),
                           prediction_text3 = " Reduction in Area (%)  =  {}".format(prediction[0][3]))

if __name__ == "__main__":
    app.run(debug=True)
Home()