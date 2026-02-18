from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Load Model Artifacts
model = pickle.load(open("static/model/model.sav", "rb"))
scaler = pickle.load(open("static/model/scaler.pkl", "rb"))
FEATURE_COLUMNS = pickle.load(open("static/model/features.pkl", "rb"))


# Routes

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/heart-disease-predictor", response_class=HTMLResponse)
def heart_disease_form(request: Request):
    return templates.TemplateResponse("heart_disease.html", {"request": request})


@app.post("/heart-disease-predictor", response_class=HTMLResponse)
def heart_disease_predict(
    request: Request,
    age: int = Form(...),
    resting_blood_pressure: int = Form(...),
    serum_cholestrol_value: int = Form(...),
    fasting_blood_sugar: int = Form(...),
    heart_rate_value: int = Form(...),
    st_depressed_value: float = Form(...),
    gender: int = Form(...),
    induced_agina: int = Form(...),
    chest_pain_type: str = Form(...),
    resting_ecg: str = Form(...),
    peak_exercise_st: str = Form(...),
):

    # Map inputs
    data = {
        "Age": age,
        "RestingBP": resting_blood_pressure,
        "Cholesterol": serum_cholestrol_value,
        "FastingBS": fasting_blood_sugar,
        "MaxHR": heart_rate_value,
        "Oldpeak": st_depressed_value,

        "Sex_M": gender,
        "ExerciseAngina_Y": induced_agina,

        "ChestPainType_ATA": 1 if chest_pain_type == "2" else 0,
        "ChestPainType_NAP": 1 if chest_pain_type == "3" else 0,
        "ChestPainType_TA":  1 if chest_pain_type == "1" else 0,

        "RestingECG_Normal": 1 if resting_ecg == "0" else 0,
        "RestingECG_ST": 1 if resting_ecg == "1" else 0,

        "ST_Slope_Flat": 1 if peak_exercise_st == "2" else 0,
        "ST_Slope_Up": 1 if peak_exercise_st == "1" else 0,
    }

    input_df = pd.DataFrame([[data.get(col, 0) for col in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)

    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    prediction_prob = int(model.predict_proba(input_scaled).max() * 100)

    results = {
        "prediction": prediction,
        "prediction-prob": prediction_prob
    }

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "results": results}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
