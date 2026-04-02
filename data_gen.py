"""
data_gen.py
Generates simulated_data.csv for initial model training.
Run this ONCE before train_model.py if you have no real data yet.
"""

import pandas as pd
import numpy as np
from random import choice, randint
import os

os.makedirs("data", exist_ok=True)

DAYS       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MENUS      = ["Regular", "Special", "Festival"]
MEAL_TIMES = ["Breakfast", "Lunch", "Dinner"]

data = []

np.random.seed(42)
for _ in range(200):
    day        = choice(DAYS)
    attendance = randint(50, 300)
    is_holiday = choice([0, 1])
    menu       = choice(MENUS)
    meal_time  = choice(MEAL_TIMES)

    # Slightly realistic: festivals → more meals, holidays → lower attendance effect
    meal_factor  = np.random.uniform(0.70, 1.10)
    waste_factor = np.random.uniform(0.05, 0.20)
    if menu == "Festival":
        meal_factor += 0.05
    if is_holiday:
        waste_factor -= 0.02   # less waste on holidays (smaller crowd, better planning)

    meals = max(0, int(attendance * meal_factor))
    waste = max(0, int(meals * max(0.03, waste_factor)))

    data.append([day, attendance, is_holiday, menu, meal_time, meals, waste])

df = pd.DataFrame(data, columns=["day", "attendance", "holiday", "menu", "meal_time", "meals", "waste"])
df.to_csv("data/simulated_data.csv", index=False)

print(f"✅ simulated_data.csv created in data/ with {len(df)} rows.")
print(df.head())