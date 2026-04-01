import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/simulated_data.csv")

# Encode categorical features
le_day = LabelEncoder()
le_menu = LabelEncoder()
le_meal_time = LabelEncoder()

df["day_encoded"] = le_day.fit_transform(df["day"])
df["menu_encoded"] = le_menu.fit_transform(df["menu"])
df["meal_time_encoded"] = le_meal_time.fit_transform(df["meal_time"])

# Features and targets
X = df[["attendance","day_encoded","holiday","menu_encoded","meal_time_encoded"]]
y_meals = df["meals"]
y_waste = df["waste"]

# Train models
meal_model = RandomForestRegressor(n_estimators=100, random_state=42)
meal_model.fit(X, y_meals)

waste_model = DecisionTreeRegressor(max_depth=5, random_state=42)
waste_model.fit(X, y_waste)

# Save models and encoders
joblib.dump(meal_model, "models/meal_model.pkl")
joblib.dump(waste_model, "models/waste_model.pkl")
joblib.dump(le_day, "models/le_day.pkl")
joblib.dump(le_menu, "models/le_menu.pkl")
joblib.dump(le_meal_time, "models/le_meal_time.pkl")

print("Models and encoders saved in models/ folder.")