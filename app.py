from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone, timedelta
import os
import io
import threading
import traceback

from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "smart-mess-analytics-data")

# ── IST TIMEZONE (UTC+5:30) ──────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    """Return current datetime in IST."""
    return datetime.now(IST)

# ── GCS INIT ─────────────────────────────────────────────
try:
    _gcs = storage.Client()
    _bucket = _gcs.bucket(GCS_BUCKET)
    GCS_OK = True
    print(f"GCS client initialised OK — bucket: {GCS_BUCKET}")
except Exception as e:
    print(f"GCS INIT FAILED: {e}")
    GCS_OK = False
    _bucket = None

# ── SCHEMAS ─────────────────────────────────────────────
ACTUALS_COLS = [
    "timestamp", "day", "meal_time", "attendance", "holiday", "menu",
    "predicted_meals", "predicted_waste", "waste_status",
    "actual_meals", "actual_waste"
]

PRED_COLS = [
    "timestamp", "day", "meal_time", "attendance", "holiday", "menu",
    "predicted_meals", "predicted_waste", "waste_status"
]

ACTUALS_FILE     = "data/actuals.csv"
PREDICTIONS_FILE = "data/predictions_log.csv"
DATA_FILE        = "data/simulated_data.csv"

# ── MODEL GLOBALS ────────────────────────────────────────
meal_model   = None
waste_model  = None
le_day       = None
le_menu      = None
le_meal_time = None
MODEL_LOADED = False

# ── CSV HELPERS ─────────────────────────────────────────
def _read_csv(filename, columns):
    if not GCS_OK:
        return pd.DataFrame(columns=columns)
    try:
        blob = _bucket.blob(filename)
        if not blob.exists():
            return pd.DataFrame(columns=columns)
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))
        return df
    except Exception as e:
        print(f"GCS READ ERROR ({filename}): {e}")
        return pd.DataFrame(columns=columns)

def _write_csv(df, filename):
    if not GCS_OK:
        return
    try:
        blob = _bucket.blob(filename)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="text/csv")
    except Exception as e:
        print(f"GCS WRITE ERROR ({filename}): {e}")

# ── MODEL LOADING ───────────────────────────────────────
def load_models():
    global meal_model, waste_model, le_day, le_menu, le_meal_time, MODEL_LOADED
    try:
        meal_model   = joblib.load("models/meal_model.pkl")
        waste_model  = joblib.load("models/waste_model.pkl")
        le_day       = joblib.load("models/le_day.pkl")
        le_menu      = joblib.load("models/le_menu.pkl")
        le_meal_time = joblib.load("models/le_meal_time.pkl")
        MODEL_LOADED = True
        print("Models loaded OK")
    except Exception as e:
        print(f"MODEL LOAD FAILED: {e}")
        traceback.print_exc()
        MODEL_LOADED = False

load_models()

# ── SAFE LABEL ENCODE ───────────────────────────────────
def safe_transform(encoder, value, fallback=0):
    try:
        return encoder.transform([value])[0]
    except Exception:
        classes = list(encoder.classes_)
        print(f"Warning: '{value}' not in encoder classes {classes}. Using fallback.")
        if classes:
            return encoder.transform([classes[0]])[0]
        return fallback

# ── RETRAIN FUNCTION ────────────────────────────────────
def retrain_and_reload():
    print("Starting retraining...")
    try:
        os.system("python train_models.py")
        print("Retraining done")
        load_models()
    except Exception as e:
        print(f"Retraining failed: {e}")
        traceback.print_exc()

# ── PREDICTION LOGGER ───────────────────────────────────
def _log_prediction(day, meal_time, attendance, holiday, menu,
                    predicted_meals, predicted_waste, status):
    try:
        timestamp = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        new_row = {
            "timestamp":       timestamp,
            "day":             day,
            "meal_time":       meal_time,
            "attendance":      attendance,
            "holiday":         holiday,
            "menu":            menu,
            "predicted_meals": predicted_meals,
            "predicted_waste": predicted_waste,
            "waste_status":    status,
            "actual_meals":    "",
            "actual_waste":    ""
        }
        df_actuals = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
        df_actuals = pd.concat([df_actuals, pd.DataFrame([new_row])], ignore_index=True)
        _write_csv(df_actuals, ACTUALS_FILE)
    except Exception as e:
        print(f"LOG PREDICTION ERROR: {e}")
        traceback.print_exc()

# ── EFFICIENCY CALCULATOR ───────────────────────────────
def _calc_overall_efficiency(df_actuals):
    """
    Efficiency = (total actual_meals served) / (total attendance) * 100
    This tells you what % of students who showed up actually got a meal.

    Priority 1: rows where actual_meals AND attendance are both logged.
    Priority 2: fallback using (predicted_meals - predicted_waste) / attendance.

    Returns (value: float|None, source: str|None)
    """
    try:
        if df_actuals.empty:
            return None, None

        df = df_actuals.copy()
        df["actual_meals"]    = pd.to_numeric(df["actual_meals"],    errors="coerce")
        df["actual_waste"]    = pd.to_numeric(df["actual_waste"],    errors="coerce")
        df["predicted_meals"] = pd.to_numeric(df["predicted_meals"], errors="coerce")
        df["predicted_waste"] = pd.to_numeric(df["predicted_waste"], errors="coerce")
        df["attendance"]      = pd.to_numeric(df["attendance"],      errors="coerce")

        # ── Priority 1: use actual_meals vs attendance ──
        df_logged = df[
            df["actual_meals"].notna() &
            df["attendance"].notna() &
            (df["attendance"] > 0)
        ]
        print(f"[EFFICIENCY] Rows with actuals: {len(df_logged)}")

        if not df_logged.empty:
            total_served     = df_logged["actual_meals"].sum()
            total_attendance = df_logged["attendance"].sum()
            eff = round(min((total_served / total_attendance) * 100, 100.0), 1)
            print(f"[EFFICIENCY] actual_meals/attendance = {eff}%")
            return eff, "actual"

        # ── Priority 2: (predicted_meals - predicted_waste) / attendance ──
        df_pred = df[
            df["predicted_meals"].notna() &
            df["attendance"].notna() &
            (df["attendance"] > 0)
        ]
        if not df_pred.empty:
            total_meals      = df_pred["predicted_meals"].sum()
            total_waste      = df_pred["predicted_waste"].fillna(0).sum()
            total_attendance = df_pred["attendance"].sum()
            consumed = max(total_meals - total_waste, 0)
            eff = round(min((consumed / total_attendance) * 100, 100.0), 1)
            print(f"[EFFICIENCY] predicted fallback = {eff}%")
            return eff, "predicted"

    except Exception as e:
        print(f"EFFICIENCY CALC ERROR: {e}")
        traceback.print_exc()

    return None, None

# ════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════
@app.route("/health")
def health():
    now = now_ist()
    return {
        "gcs_ok":       GCS_OK,
        "model_loaded": MODEL_LOADED,
        "bucket":       GCS_BUCKET,
        "ist_time":     now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "status":       "ok"
    }

# ════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def dashboard():
    try:
        now  = now_ist()                        # ✅ IST time
        day  = now.strftime("%A")
        hour = now.hour

        # Meal time based on IST hour
        if 7 <= hour <= 10:
            meal_time = "Breakfast"
        elif 11 <= hour <= 14:
            meal_time = "Lunch"
        elif 18 <= hour <= 21:
            meal_time = "Dinner"
        else:
            meal_time = "Lunch"

        attendance = 150
        is_holiday = "No"
        menu_type  = "Regular"

        if request.method == "POST":
            try:
                attendance = int(request.form.get("attendance", 150))
            except (ValueError, TypeError):
                attendance = 150
            is_holiday = request.form.get("holiday", "No")
            menu_type  = request.form.get("menu", "Regular")
            meal_time  = request.form.get("meal_time", meal_time)
            day        = request.form.get("day", day)

        holiday_flag = 1 if is_holiday == "Yes" else 0

        predicted_meals = 100
        predicted_waste = 10

        if MODEL_LOADED:
            try:
                X_input = np.array([[
                    attendance,
                    safe_transform(le_day, day),
                    holiday_flag,
                    safe_transform(le_menu, menu_type),
                    safe_transform(le_meal_time, meal_time)
                ]])
                predicted_meals = max(0, int(meal_model.predict(X_input)[0]))
                predicted_waste = max(0, int(waste_model.predict(X_input)[0]))
            except Exception as e:
                print(f"PREDICTION ERROR: {e}")
                traceback.print_exc()
                predicted_meals = 100
                predicted_waste = 10

        waste_ratio = predicted_waste / predicted_meals if predicted_meals > 0 else 0

        if waste_ratio > 0.25:
            status = "High ⚠️"
        elif waste_ratio > 0.15:
            status = "Moderate ⚡"
        else:
            status = "Low ✅"

        if request.method == "POST":
            _log_prediction(day, meal_time, attendance, is_holiday, menu_type,
                            predicted_meals, predicted_waste, status)

        df_actuals      = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
        actuals_records = df_actuals.tail(50).to_dict(orient="records")

        overall_efficiency, efficiency_source = _calc_overall_efficiency(df_actuals)

        df_past         = _read_csv(DATA_FILE, columns=[])
        df_past_records = df_past.to_dict(orient="records") if not df_past.empty else []

        return render_template(
            "index.html",
            attendance=attendance,
            predicted_meals=predicted_meals,
            predicted_waste=predicted_waste,
            status=status,
            day=day,
            meal_time=meal_time,
            holiday=is_holiday,
            menu_type=menu_type,
            df_past=df_past_records,
            actuals=actuals_records,
            model_loaded=MODEL_LOADED,
            gcs_ok=GCS_OK,
            overall_efficiency=overall_efficiency,
            efficiency_source=efficiency_source,
            ist_time=now.strftime("%d %b %Y, %I:%M %p IST")
        )

    except Exception as e:
        print(f"DASHBOARD ERROR: {e}")
        traceback.print_exc()
        return render_template(
            "index.html",
            attendance=150,
            predicted_meals=100,
            predicted_waste=10,
            status="Low ✅",
            day=now_ist().strftime("%A"),
            meal_time="Lunch",
            holiday="No",
            menu_type="Regular",
            df_past=[],
            actuals=[],
            model_loaded=False,
            gcs_ok=GCS_OK,
            overall_efficiency=None,
            efficiency_source=None,
            ist_time=now_ist().strftime("%d %b %Y, %I:%M %p IST"),
            error=str(e)
        )

# ════════════════════════════════════════════════════════
# LOG ACTUAL
# ════════════════════════════════════════════════════════
@app.route("/log_actual", methods=["POST"])
def log_actual():
    try:
        timestamp    = request.form.get("timestamp")
        actual_meals = request.form.get("actual_meals", type=int)
        actual_waste = request.form.get("actual_waste", type=int)

        if not timestamp:
            print("log_actual: no timestamp provided")
            return redirect(url_for("dashboard"))

        df_actuals = _read_csv(ACTUALS_FILE, ACTUALS_COLS)

        if df_actuals.empty:
            print("log_actual: actuals CSV is empty or unavailable")
            return redirect(url_for("dashboard"))

        mask = df_actuals["timestamp"] == timestamp

        if mask.any():
            df_actuals.loc[mask, "actual_meals"] = actual_meals
            df_actuals.loc[mask, "actual_waste"] = actual_waste
            _write_csv(df_actuals, ACTUALS_FILE)

            matched = df_actuals[mask].iloc[0]

            new_row = {
                "day":        matched["day"],
                "attendance": matched["attendance"],
                "holiday":    1 if str(matched["holiday"]).strip().lower() == "yes" else 0,
                "menu":       matched["menu"],
                "meal_time":  matched["meal_time"],
                "meals":      actual_meals,
                "waste":      actual_waste
            }

            df_sim = _read_csv(DATA_FILE, [])
            df_sim = pd.concat([df_sim, pd.DataFrame([new_row])], ignore_index=True)
            _write_csv(df_sim, DATA_FILE)

            threading.Thread(target=retrain_and_reload, daemon=True).start()
        else:
            print(f"log_actual: timestamp '{timestamp}' not found in actuals")

    except Exception as e:
        print(f"LOG ACTUAL ERROR: {e}")
        traceback.print_exc()

    return redirect(url_for("dashboard"))

# ════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)