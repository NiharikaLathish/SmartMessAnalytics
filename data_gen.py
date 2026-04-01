import pandas as pd
import numpy as np
from random import choice, randint
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
menus = ["Regular", "Special", "Festival"]
meal_times = ["Breakfast", "Lunch", "Dinner"]

data = []

for _ in range(200):
    day = choice(days)
    attendance = randint(50, 300)
    is_holiday = choice([0,1])
    menu = choice(menus)
    meals = int(attendance * np.random.uniform(0.7,1.1))
    waste = int(meals * np.random.uniform(0.05,0.2))
    meal_time = choice(meal_times)
    data.append([day, attendance, is_holiday, menu, meal_time, meals, waste])

df = pd.DataFrame(data, columns=["day","attendance","holiday","menu","meal_time","meals","waste"])
df.to_csv("data/simulated_data.csv", index=False)
print("simulated_data.csv created in data/ folder with 200 rows!")