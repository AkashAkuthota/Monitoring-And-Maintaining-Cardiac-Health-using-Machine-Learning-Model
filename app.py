from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ===============================
# Load Model Artifacts
# ===============================
model = pickle.load(open("static/model/model.sav", "rb"))
scaler = pickle.load(open("static/model/scaler.pkl", "rb"))
FEATURE_COLUMNS = pickle.load(open("static/model/features.pkl", "rb"))

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/heart-disease-predictor", methods=["GET", "POST"])
def heartDiseasePredictor():
    if request.method == "POST":
        form = request.form.to_dict()

        # -------------------------------
        # Map inputs
        # -------------------------------
        data = {
            "Age": int(form["age"]),
            "RestingBP": int(form["resting-blood-pressure"]),
            "Cholesterol": int(form["serum-cholestrol-value"]),
            "FastingBS": int(form["fasting-blood-sugar"]),
            "MaxHR": int(form["heart-rate-value"]),
            "Oldpeak": float(form["st-depressed-value"]),

            "Sex_M": int(form["gender"]),
            "ExerciseAngina_Y": int(form["induced-agina"]),

            "ChestPainType_ATA": 1 if form["chest-pain-type"] == "2" else 0,
            "ChestPainType_NAP": 1 if form["chest-pain-type"] == "3" else 0,
            "ChestPainType_TA":  1 if form["chest-pain-type"] == "1" else 0,

            "RestingECG_Normal": 1 if form["resting-ecg"] == "0" else 0,
            "RestingECG_ST": 1 if form["resting-ecg"] == "1" else 0,

            "ST_Slope_Flat": 1 if form["peak-exercise-st"] == "2" else 0,
            "ST_Slope_Up": 1 if form["peak-exercise-st"] == "1" else 0,
        }

        input_df = pd.DataFrame([[data.get(col, 0) for col in FEATURE_COLUMNS]],
                                columns=FEATURE_COLUMNS)

        input_scaled = scaler.transform(input_df)

        prediction = int(model.predict(input_scaled)[0])
        prediction_prob = int(model.predict_proba(input_scaled).max() * 100)

        form["prediction"] = prediction
        form["prediction-prob"] = prediction_prob

        return render_template("result.html", results=form)

    return render_template("heart_disease.html")

if __name__ == "__main__":
    app.run()
