# ETo Forecast

Predicting next-day reference evapotranspiration (ETo) using an LSTM neural network, with a live Streamlit app that converts predictions into irrigation recommendations using the FAO 56 crop water requirement formula.

**[Live App](https://eto-forecast.streamlit.app)** · CS 4210 Machine Learning and Its Applications · Cal Poly Pomona · Spring 2026

## Overview

Reference evapotranspiration (ETo) is the key input to irrigation planning, but it can only be calculated from observed weather. It cannot be forecasted. Irrigators need tomorrow's ETo today to schedule watering in advance.

This project trains an LSTM with a learned attention mechanism on 22 years of daily weather data from CIMIS Station 44 (UC Riverside) to predict next-day ETo. The model achieves a test MAE of 0.636 mm/day on a held-out 2024-2026 test set.

The Streamlit app fetches live weather data from CIMIS, runs the model, and produces a daily irrigation recommendation for any of 137 FAO 56 crops, with crop coefficients adjusted for local climate conditions (FAO 56 Eq. 62) and precipitation forecasts from the National Weather Service API.

## Project Structure

```
eto-forecast/
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
│   └── secrets.toml         # API keys (not committed, see setup)
├── secrets.toml.example      # Template for secrets
├── app.py                    # Streamlit app (deployment + inference)
├── Final_Project.ipynb       # Training notebook (data processing, model training, evaluation)
├── daily.csv                 # CIMIS Station 44 data, 2000-2026 (9,593 records)
├── eto_lstm_best.pth         # Best model weights (saved by early stopping)
├── eto_lstm_model.pth        # Final model weights
├── eto_scaler.pkl            # Fitted MinMaxScaler
├── eto_model_config.pkl      # Model hyperparameters, column names, climate stats
├── requirements.txt          # Python dependencies
└── README.md
```

## Run the Notebook

The training notebook runs in Google Colab with GPU.

1. Open `Final_Project.ipynb` in Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Run All (Runtime > Run all)

The notebook produces the model files (`eto_lstm_best.pth`, `eto_scaler.pkl`, `eto_model_config.pkl`) used by the Streamlit app.

## Run the App Locally

```bash
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your [CIMIS API key](https://et.water.ca.gov/Home/Register):

```toml
CIMIS_API_KEY = "your-key-here"
```

Then run:

```bash
streamlit run app.py
```

## Model

**Architecture:** 2-layer LSTM with learned attention mechanism

| Parameter | Value |
| :--- | :--- |
| Input | 14 days x 12 weather features |
| Hidden size | 128 |
| Layers | 2 |
| Loss function | L1Loss (MAE) |
| Optimizer | Adam |
| Training data | 2000-2021 (8,022 sequences) |
| Validation data | 2022-2023 |
| Test data | 2024-2026 |
| Test MAE | 0.636 mm/day |
| Test RMSE | 0.932 mm/day |

Seven model variants were evaluated. The combination of attention with L1Loss achieved the best result, a 7.2% improvement over the baseline LSTM.

## Data

Daily weather observations from [CIMIS](https://cimis.water.ca.gov) Station 44 (UC Riverside), January 2000 to April 2026.

**Features (12):** Solar Radiation, Avg Vapor Pressure, Max/Min/Avg Air Temperature, Max/Min/Avg Relative Humidity, Dew Point, Avg Wind Speed, Avg Soil Temperature, Day of Year

**Target:** ETo (mm/day)

**Split:** Chronological (no shuffling). Train: 2000-2021, Validation: 2022-2023, Test: 2024-2026.

## App Features

- Live weather data from CIMIS API
- Next-day ETo prediction using trained LSTM
- 137 FAO 56 crops with searchable selection (synonym-aware via pyfao56)
- Crop coefficients adjusted for local climate (FAO 56 Eq. 62)
- Precipitation forecast from National Weather Service API
- Irrigation efficiency by method (drip, sprinkler, flood)
- Attention weight visualization
- Live prediction accuracy backtesting

## References

- Roy, D.K. et al. (2022). "Daily Prediction and Multi-Step Forward Forecasting of Reference Evapotranspiration Using LSTM and Bi-LSTM Models." *Agronomy*, 12(3), 594. [doi:10.3390/agronomy12030594](https://doi.org/10.3390/agronomy12030594)
- Allen, R.G. et al. (1998). *Crop Evapotranspiration*. FAO Irrigation and Drainage Paper 56.
- Jia, W. et al. (2023). "Daily reference evapotranspiration prediction based on the hybrid PSO-LSTM model." *PLOS One*. [doi:10.1371/journal.pone.0281478](https://doi.org/10.1371/journal.pone.0281478)
- Li, Z. et al. (2024). "Prediction of reference crop evapotranspiration based on improved CNN and LSTM models." *J. Hydrol.* [doi:10.1016/j.jhydrol.2024.132223](https://doi.org/10.1016/j.jhydrol.2024.132223)

## Author

Lindsay Kislingbury
CS 4210 Machine Learning and Its Applications
Professor Hao Ji, Cal Poly Pomona, Spring 2026