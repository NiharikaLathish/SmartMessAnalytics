from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os
import io

from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "smart-mess-analytics-data")

# ── Single GCS client created once at startup ─────────────────────
try:
    _gcs = storage.Client()
    _bucket = _gcs.bucket(GCS_BUCKET)
    GCS_OK = True
    print(f"GCS client initialised OK — bucket: {GCS_BUCKET}")
except Exception as e:
    print(f"GCS INIT FAILED: {e}")
    GCS_OK = False
    _bucket = None

# ── Column schemas ────────────────────────────────────────────────
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


def _read_csv(filename, columns):
    """Read CSV from GCS. Returns empty DataFrame on any failure."""
    if not GCS_OK:
        return pd.DataFrame(columns=columns)
    try:
        blob = _bucket.blob(filename)
        if not blob.exists():
            print(f"GCS: {filename} does not exist yet — returning empty DataFrame")
            return pd.DataFrame(columns=columns)
        data = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        print(f"GCS READ ERROR ({filename}): {e}")
        return pd.DataFrame(columns=columns)


def _write_csv(df, filename):
    """Write DataFrame as CSV to GCS."""
    if not GCS_OK:
        print(f"GCS not available — skipping write to {filename}")
        return
    try:
        blob = _bucket.blob(filename)
        buf  = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="text/csv")
        print(f"GCS WRITE OK: {filename} ({len(df)} rows)")
    except Exception as e:
        print(f"GCS WRITE ERROR ({filename}): {e}")


# ── SAFE Model Loading ────────────────────────────────────────────
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
    MODEL_LOADED = False

# ── Load past data once at startup ───────────────────────────────
# df_past = _read_csv(DATA_FILE, columns=[])
# print(f"Loaded {len(df_past)} rows from simulated_data.csv")

# df_past loaded lazily per-request, not at startup
# ══════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def dashboard():

    now  = datetime.now()
    day  = now.strftime("%A")
    hour = now.hour

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
        attendance = int(request.form.get("attendance", 150))
        is_holiday = request.form.get("holiday", "No")
        menu_type  = request.form.get("menu", "Regular")
        meal_time  = request.form.get("meal_time", meal_time)
        day        = request.form.get("day", day)

    holiday_flag = 1 if is_holiday == "Yes" else 0

    # ── Prediction ────────────────────────────────────────────────
    if MODEL_LOADED:
        try:
            X_input = np.array([[
                attendance,
                le_day.transform([day])[0],
                holiday_flag,
                le_menu.transform([menu_type])[0],
                le_meal_time.transform([meal_time])[0]
            ]])
            predicted_meals = max(0, int(meal_model.predict(X_input)[0]))
            predicted_waste = max(0, int(waste_model.predict(X_input)[0]))
        except Exception as e:
            print(f"PREDICTION ERROR: {e}")
            predicted_meals = 100
            predicted_waste = 10
    else:
        predicted_meals = 100
        predicted_waste = 10

    # ── Waste Status ──────────────────────────────────────────────
    waste_ratio = predicted_waste / predicted_meals if predicted_meals > 0 else 0
    if waste_ratio > 0.25:
        status = "High ⚠️"
    elif waste_ratio > 0.15:
        status = "Moderate ⚡"
    else:
        status = "Low ✅"

    # ── Log prediction on POST ────────────────────────────────────
    if request.method == "POST":
        _log_prediction(day, meal_time, attendance, is_holiday, menu_type,
                        predicted_meals, predicted_waste, status)

    # ── Load actuals fresh from GCS ───────────────────────────────
    df_actuals      = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
    actuals_records = df_actuals.tail(50).to_dict(orient="records")

    efficiency = None
    filled = df_actuals.dropna(subset=["actual_meals", "actual_waste"])
    if not filled.empty:
        total_prepared = filled["actual_meals"].sum()
        total_wasted   = filled["actual_waste"].sum()
        if total_prepared > 0:
            efficiency = round(100 - (total_wasted / total_prepared * 100), 1)

    df_past = _read_csv(DATA_FILE, columns=[])

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
        df_past=df_past.to_dict(orient="records"),
        actuals=actuals_records,
        efficiency=efficiency,
    )


# ══════════════════════════════════════════════════════════════════
# LOG ACTUAL
# ══════════════════════════════════════════════════════════════════
@app.route("/log_actual", methods=["POST"])
def log_actual():

    timestamp    = request.form.get("timestamp")
    actual_meals = request.form.get("actual_meals", type=int)
    actual_waste = request.form.get("actual_waste", type=int)

    print(f"LOG ACTUAL called: ts={timestamp}, meals={actual_meals}, waste={actual_waste}")

    df_actuals = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
    mask = df_actuals["timestamp"] == timestamp

    if mask.any():
        df_actuals.loc[mask, "actual_meals"] = actual_meals
        df_actuals.loc[mask, "actual_waste"] = actual_waste
        _write_csv(df_actuals, ACTUALS_FILE)

        # Append verified row to simulated_data.csv for retraining
        matched = df_actuals[mask].iloc[0]
        new_sim_row = {
            "day":        matched["day"],
            "attendance": matched["attendance"],
            "holiday":    1 if str(matched["holiday"]).strip() == "Yes" else 0,
            "menu":       matched["menu"],
            "meal_time":  matched["meal_time"],
            "meals":      actual_meals,
            "waste":      actual_waste,
        }
        df_sim = _read_csv(DATA_FILE, columns=[])
        df_sim = pd.concat([df_sim, pd.DataFrame([new_sim_row])], ignore_index=True)
        _write_csv(df_sim, DATA_FILE)
        print(f"Appended real row to simulated_data.csv — now {len(df_sim)} rows")

    else:
        print(f"WARNING: timestamp {timestamp} not found in actuals — saving fallback row")
        new_row = {col: None for col in ACTUALS_COLS}
        new_row["timestamp"]    = timestamp
        new_row["actual_meals"] = actual_meals
        new_row["actual_waste"] = actual_waste
        df_actuals = pd.concat([df_actuals, pd.DataFrame([new_row])], ignore_index=True)
        _write_csv(df_actuals, ACTUALS_FILE)

    return redirect(url_for("dashboard"))


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _log_prediction(day, meal_time, attendance, holiday, menu,
                    predicted_meals, predicted_waste, waste_status):

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"LOG PREDICTION: {ts} | {day} | {meal_time} | att={attendance}")

    row = {
        "timestamp":       ts,
        "day":             day,
        "meal_time":       meal_time,
        "attendance":      attendance,
        "holiday":         holiday,
        "menu":            menu,
        "predicted_meals": predicted_meals,
        "predicted_waste": predicted_waste,
        "waste_status":    waste_status,
        "actual_meals":    None,
        "actual_waste":    None,
    }

    # ── Write to predictions_log ──────────────────────────────────
    df_pred = _read_csv(PREDICTIONS_FILE, PRED_COLS)
    df_pred = pd.concat([df_pred, pd.DataFrame([row])], ignore_index=True)
    _write_csv(df_pred, PREDICTIONS_FILE)

    # ── Write to actuals — ONE read+write only ────────────────────
    df_act = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
    df_act = pd.concat([df_act, pd.DataFrame([row])], ignore_index=True)
    _write_csv(df_act, ACTUALS_FILE)
    print(f"actuals.csv now has {len(df_act)} rows")


# ══════════════════════════════════════════════════════════════════
# DATA PREVIEW
# ══════════════════════════════════════════════════════════════════
@app.route("/data-preview")
def data_preview():
    df_sim  = _read_csv(DATA_FILE, columns=[])
    df_act  = _read_csv(ACTUALS_FILE, ACTUALS_COLS)
    df_pred = _read_csv(PREDICTIONS_FILE, PRED_COLS)
    return f"""
    <html><body style="font-family:sans-serif;padding:24px;">
    <h2>GCS status: {'✅ Connected' if GCS_OK else '❌ Not connected'}</h2>
    <h2>simulated_data.csv — {len(df_sim)} total rows</h2>
    {df_sim.tail(10).to_html()}
    <hr/>
    <h2>actuals.csv — {len(df_act)} total rows</h2>
    {df_act.tail(10).to_html()}
    <hr/>
    <h2>predictions_log.csv — {len(df_pred)} total rows</h2>
    {df_pred.tail(10).to_html()}
    </body></html>
    """


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)