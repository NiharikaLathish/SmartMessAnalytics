# 🍽️ Foodley — Smart Mess Analytics

> Predicting how much food to prepare, so nothing goes to waste.

Foodley is a machine learning–powered web application that helps college mess facilities optimize food preparation quantities. By analyzing historical wastage data, student attendance patterns, weekdays, and holidays, it predicts the right amount of food to prepare — reducing waste and improving efficiency.

---

## 🚀 Features

- **Smart Quantity Prediction** — ML model trained on attendance, past wastage, weekday/holiday patterns
- **Data Generation** — Synthetic data pipeline (`data_gen.py`) for training and testing
- **Web Dashboard** — Flask-based UI for inputting parameters and viewing predictions
- **Dockerized** — Ready to deploy anywhere with a single command

---

## 🧠 How It Works

1. Input variables: student attendance count, day of week, holiday flag, historical wastage
2. Trained model processes inputs and predicts the optimal food quantity to prepare
3. Result is displayed via the web interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn / custom models (`train_models.py`) |
| Data | Synthetic data generation (`data_gen.py`) |
| Frontend | HTML/CSS (Jinja2 templates) |
| Deployment | Docker |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repo
git clone https://github.com/NiharikaLathish/SmartMessAnalytics.git
cd SmartMessAnalytics

# Install dependencies
pip install -r requirements.txt

# Train the models
python train_models.py

# Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

### Docker Setup

```bash
docker build -t foodley .
docker run -p 5000:5000 foodley
```

---

## 📁 Project Structure

```
SmartMessAnalytics/
├── data/               # Training and input data
├── models/             # Saved ML model files
├── templates/          # Flask HTML templates
├── app.py              # Main Flask application
├── train_models.py     # Model training script
├── data_gen.py         # Synthetic data generation
├── requirements.txt    # Python dependencies
└── Dockerfile          # Container configuration
```

---

## 👥 Contributors

| Name | GitHub |
|---|---|
| Niharika Lathish | [@NiharikaLathish](https://github.com/NiharikaLathish) |
| Maneesh Ari | [@arinova2701](https://github.com/arinova2701) |
| Roshan Prabu | — |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built at VIT Vellore. Stay Curious. Build Relentlessly.*
