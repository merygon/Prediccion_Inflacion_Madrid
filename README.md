# 📈 Spain Inflation Prediction System

An end-to-end **ML pipeline** that downloads official INE data, trains multiple forecasting models, and generates 12-month inflation predictions with confidence intervals and a full PDF report — fully automated and reproducible.

---

## 🎯 What It Does

1. **Fetches** CPI time series directly from the INE (Instituto Nacional de Estadística) API
2. **Cleans & engineers features** — outlier detection, lag features, moving averages, seasonality decomposition
3. **Trains 3 models** — ARIMA (baseline), Random Forest, LSTM — and auto-selects the best performer
4. **Generates forecasts** for the next 12 months with confidence intervals
5. **Produces a full PDF report** with charts and economic analysis

---

## 📊 Sample Output

```
predictions.csv
fecha,predicted_inflation,confidence_lower,confidence_upper,model_used
2025-01-01,2.45,1.89,3.01,LSTM
2025-02-01,2.52,1.95,3.09,LSTM
...
```

Plus auto-generated visualizations: historical trends, model comparison, confidence interval bands, seasonal decomposition.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10–3.12 |
| Time Series | statsmodels (ARIMA) |
| ML | scikit-learn (Random Forest) |
| Deep Learning | TensorFlow/Keras (LSTM) — optional |
| Data Source | INE WSTempus API |
| Reporting | PDF generation + matplotlib |
| Config | YAML |

---

## ⚙️ Installation

```bash
git clone https://github.com/merygon/Prediccion_Inflacion_Madrid.git
cd Prediccion_Inflacion_Madrid

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Run

```bash
python src/main.py
```

Pipeline sequence: `download → clean → feature engineering → train → forecast → PDF report`

Logs saved to `logs/inflation_prediction.log`

---

## ⚙️ Configuration

Edit `config/config.yaml` to adjust dates, models, and INE series IDs:

```yaml
data:
  start_date: "2002-01-01"
  end_date: "2024-12-31"

models:
  arima:
    max_p: 5
    max_d: 2
    max_q: 5
  random_forest:
    n_estimators: 100
  lstm:
    epochs: 100
    batch_size: 32

prediction:
  horizon_months: 12
  confidence_level: 0.95
```

To find INE series IDs: search "IPC General" on [WSTempus](https://servicios.ine.es/wstempus), open the series and use the numeric ID at the end of the URL.

---

## 🧩 Pipeline Architecture

```
src/
├── main.py                 # Pipeline orchestrator
├── ine_extractor.py        # INE API downloader
├── data_cleaner.py         # Cleaning, outlier detection, normalization
├── feature_engineering.py  # Lag features, moving averages, seasonality
├── model_trainer.py        # ARIMA + RF + LSTM training + model selection
├── predictor.py            # Forecasting + confidence intervals
└── report_generator.py     # Charts + PDF report generation
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

---

## 📤 Outputs

```
reports/
├── predictions.csv / predictions.json
├── technical_report.pdf
└── visualizations/
    ├── inflation_trends.png
    ├── model_comparison.png
    ├── confidence_intervals.png
    └── seasonal_decomposition.png
models/
├── arima_model.pkl
├── random_forest.pkl
└── lstm_model.pkl         # if enabled
```

---

## 📜 Data Source

Official data from [Instituto Nacional de Estadística (INE)](https://www.ine.es/) via WSTempus API. Usage governed by INE's terms of service.

---

## 📚 Context

Built as a final project for the **AI & Data Science** curriculum — Universidad Pontificia Comillas (ICAI), 2025.

---

## 👩‍💻 Author

María González Gómez · [GitHub](https://github.com/merygon)
