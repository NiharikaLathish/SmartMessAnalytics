"""
train_model.py
Run this script whenever you want to (re)train the models.
It reads data/simulated_data.csv, which grows over time as real
actual values are logged through the dashboard.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────
DATA_PATH   = "data/simulated_data.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print(f"Training on {len(df)} rows from {DATA_PATH}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Drop rows with missing targets
df = df.dropna(subset=["meals", "waste"])

# ── Encode categorical features ────────────────────────────────────
le_day       = LabelEncoder()
le_menu      = LabelEncoder()
le_meal_time = LabelEncoder()

df["day_encoded"]       = le_day.fit_transform(df["day"])
df["menu_encoded"]      = le_menu.fit_transform(df["menu"])
df["meal_time_encoded"] = le_meal_time.fit_transform(df["meal_time"])

# ── Features and targets ───────────────────────────────────────────
FEATURES = ["attendance", "day_encoded", "holiday", "menu_encoded", "meal_time_encoded"]
X        = df[FEATURES]
y_meals  = df["meals"]
y_waste  = df["waste"]

# ── Train / test split ─────────────────────────────────────────────
X_train, X_test, ym_train, ym_test, yw_train, yw_test = train_test_split(
    X, y_meals, y_waste, test_size=0.2, random_state=42
)

# ── Meal model: Random Forest ──────────────────────────────────────
meal_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
meal_model.fit(X_train, ym_train)
ym_pred = meal_model.predict(X_test)
print("── Meal Model (RandomForest) ──────────────────────")
print(f"  MAE : {mean_absolute_error(ym_test, ym_pred):.2f}")
print(f"  R²  : {r2_score(ym_test, ym_pred):.4f}\n")

# ── Waste model: Gradient Boosting (better than plain DecisionTree) ─
waste_model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
waste_model.fit(X_train, yw_train)
yw_pred = waste_model.predict(X_test)
print("── Waste Model (GradientBoosting) ─────────────────")
print(f"  MAE : {mean_absolute_error(yw_test, yw_pred):.2f}")
print(f"  R²  : {r2_score(yw_test, yw_pred):.4f}\n")

# ── Save everything ────────────────────────────────────────────────
joblib.dump(meal_model,   f"{MODELS_DIR}/meal_model.pkl")
joblib.dump(waste_model,  f"{MODELS_DIR}/waste_model.pkl")
joblib.dump(le_day,       f"{MODELS_DIR}/le_day.pkl")
joblib.dump(le_menu,      f"{MODELS_DIR}/le_menu.pkl")
joblib.dump(le_meal_time, f"{MODELS_DIR}/le_meal_time.pkl")

print("✅ Models and encoders saved to models/ folder.")
print(f"   Known days       : {list(le_day.classes_)}")
print(f"   Known menus      : {list(le_menu.classes_)}")
print(f"   Known meal times : {list(le_meal_time.classes_)}")