from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# ---- Load Models and Encoders ----
meal_model = joblib.load("models/meal_model.pkl")
waste_model = joblib.load("models/waste_model.pkl")
le_day = joblib.load("models/le_day.pkl")
le_menu = joblib.load("models/le_menu.pkl")
le_meal_time = joblib.load("models/le_meal_time.pkl")

# ---- Load Simulated Past Data ----
df_past = pd.read_csv("data/simulated_data.csv")

@app.route("/", methods=["GET", "POST"])
def dashboard():
    # Auto-detect day & meal time
    now = datetime.now()
    day = now.strftime("%A")
    hour = now.hour
    if 7 <= hour <= 9:
        meal_time = "Breakfast"
    elif 12 <= hour <= 14:
        meal_time = "Lunch"
    elif 19 <= hour <= 21:
        meal_time = "Dinner"
    else:
        meal_time = "Lunch"  # default fallback

    # Default inputs
    attendance = 150
    is_holiday = "No"
    menu_type = "Regular"

    # If form submitted
    if request.method == "POST":
        attendance = int(request.form.get("attendance", 150))
        is_holiday = request.form.get("holiday", "No")
        menu_type = request.form.get("menu", "Regular")

    holiday_flag = 1 if is_holiday == "Yes" else 0

    # Encode features
    X_input = np.array([[attendance,
                         le_day.transform([day])[0],
                         holiday_flag,
                         le_menu.transform([menu_type])[0],
                         le_meal_time.transform([meal_time])[0]]])

    predicted_meals = int(meal_model.predict(X_input)[0])
    predicted_waste = int(waste_model.predict(X_input)[0])

    # Waste alert logic
    if predicted_waste > predicted_meals * 0.25:
        status = "High ⚠️"
    elif predicted_waste > predicted_meals * 0.15:
        status = "Moderate ⚡"
    else:
        status = "Low ✅"

    return render_template("index.html",
                           attendance=attendance,
                           predicted_meals=predicted_meals,
                           predicted_waste=predicted_waste,
                           status=status,
                           day=day,
                           meal_time=meal_time,
                           holiday=is_holiday,
                           menu_type=menu_type,
                           df_past=df_past.to_dict(orient="records"))

