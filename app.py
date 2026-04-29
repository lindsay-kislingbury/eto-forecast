import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import requests
import re
import altair as alt
from datetime import date, datetime, timedelta
from pyfao56 import tools

st.set_page_config(page_title="ETo Forecast", layout="wide")

st.markdown(
    "<style>button[data-baseweb='tab'] p {font-size: 1.1rem;}</style>",
    unsafe_allow_html=True,
)


# --- Model Definition (must match training notebook) ---
class EToLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        out = torch.sum(attn_weights * out, dim=1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)

    def predict_with_attention(self, x):
        """Forward pass that also returns attention weights."""
        out, (h_n, c_n) = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1), attn_weights.squeeze(-1)


# --- Load Model, Scaler, Config ---
@st.cache_resource
def load_model():
    config = pickle.load(open("eto_model_config.pkl", "rb"))
    scaler = pickle.load(open("eto_scaler.pkl", "rb"))
    model = EToLSTMAttention(
        config["input_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    )
    model.load_state_dict(torch.load("eto_lstm_best.pth", map_location="cpu"))
    model.eval()
    return model, scaler, config


model, scaler, config = load_model()

# Station 44 long-term climate averages for FAO 56 Eq. 62
MEAN_U2 = config.get("mean_wind_speed", 1.76)
MEAN_RH_MIN = config.get("mean_rh_min", 31.3)


# --- Load FAO 56 Tables ---
@st.cache_resource
def load_fao_tables():
    return tools.FAO56Tables()


fao_tables = load_fao_tables()
crops_df = fao_tables.table12


# ============================================================
# Crop Group Definitions
# ============================================================

CROP_GROUPS = {
    "Citrus": {
        "type": "ground_cover",
        "secondary_label": "Canopy coverage",
        "secondary_options": ["70%", "50%", "20%"],
        "lookup": {
            ("None", "70%"): 119,
            ("None", "50%"): 120,
            ("None", "20%"): 121,
            ("Active", "70%"): 122,
            ("Active", "50%"): 123,
            ("Active", "20%"): 124,
        },
    },
    "Apples / Cherries / Pears": {
        "type": "ground_cover",
        "secondary_label": "Frost conditions",
        "secondary_options": ["Killing frost", "No frost"],
        "lookup": {
            ("None", "Killing frost"): 110,
            ("None", "No frost"): 111,
            ("Active", "Killing frost"): 112,
            ("Active", "No frost"): 113,
        },
    },
    "Apricots / Peaches / Stone Fruit": {
        "type": "ground_cover",
        "secondary_label": "Frost conditions",
        "secondary_options": ["Killing frost", "No frost"],
        "lookup": {
            ("None", "Killing frost"): 114,
            ("None", "No frost"): 115,
            ("Active", "Killing frost"): 116,
            ("Active", "No frost"): 117,
        },
    },
    "Maize (Corn)": {
        "type": "variants",
        "options": {
            "Field grain, 18% moisture": 70,
            "Field grain, harvested wet": 71,
            "Sweet corn, fresh": 72,
            "Sweet corn, dry": 73,
        },
    },
    "Onions": {"type": "variants", "options": {"Dry": 9, "Green": 10, "Seed": 11}},
    "Cucumber": {
        "type": "variants",
        "options": {"Fresh market": 20, "Machine harvest": 21},
    },
    "Beans": {"type": "variants", "options": {"Green": 36, "Dry / Pulses": 37}},
    "Faba Bean (Broad)": {"type": "variants", "options": {"Fresh": 39, "Dry seed": 40}},
    "Cowpeas (Green Gram)": {
        "type": "variants",
        "options": {"Harvested fresh": 42, "Harvested dry": 43},
    },
    "Peas": {"type": "variants", "options": {"Fresh": 46, "Dry seed": 47}},
    "Wheat": {
        "type": "variants",
        "options": {
            "Spring": 67,
            "Winter, frozen soils": 68,
            "Winter, non-frozen soils": 69,
        },
    },
    "Sorghum": {"type": "variants", "options": {"Grain": 75, "Sweet": 76}},
    "Cassava": {"type": "variants", "options": {"Year 1": 28, "Year 2": 29}},
    "Banana": {"type": "variants", "options": {"First year": 93, "Second year": 94}},
    "Alfalfa": {
        "type": "variants",
        "options": {
            "Hay, averaged cutting": 78,
            "Hay, individual cutting periods": 79,
            "For seed": 80,
        },
    },
    "Bermuda": {
        "type": "variants",
        "options": {"Hay, averaged cutting": 81, "Spring crop for seed": 82},
    },
    "Clover (Berseem)": {
        "type": "variants",
        "options": {"Hay, averaged cutting": 83, "Hay, individual cutting periods": 84},
    },
    "Sudangrass": {
        "type": "variants",
        "options": {"Hay, averaged cutting": 86, "Individual cutting periods": 87},
    },
    "Turfgrass": {
        "type": "variants",
        "options": {"Cool season": 90, "Warm season": 91},
    },
    "Coffee": {"type": "variants", "options": {"Bare ground": 96, "With weeds": 97}},
    "Pineapple": {
        "type": "variants",
        "options": {"Bare soil": 100, "With grass cover": 101},
    },
    "Tea": {"type": "variants", "options": {"Non-shaded": 103, "Shaded": 104}},
    "Grapes": {"type": "variants", "options": {"Table / Raisin": 106, "Wine": 107}},
    "Grazing Pasture": {
        "type": "variants",
        "options": {"Rotated": 88, "Extensive": 89},
    },
    "Cattails / Bulrushes": {
        "type": "variants",
        "options": {"Killing frost": 130, "No frost": 131},
    },
    "Reed Swamp": {
        "type": "variants",
        "options": {"Standing water": 133, "Moist soil": 134},
    },
    "Open Water": {
        "type": "variants",
        "options": {"< 2m depth / subhumid": 135, "> 5m depth / temperate": 136},
    },
}

_SIMPLE = {
    "Small Vegetables (general)": 0,
    "Broccoli": 1,
    "Brussels Sprouts": 2,
    "Cabbage": 3,
    "Carrots": 4,
    "Cauliflower": 5,
    "Celery": 6,
    "Garlic": 7,
    "Lettuce": 8,
    "Spinach": 12,
    "Radish": 13,
    "Vegetables — Solanaceae (general)": 14,
    "Eggplant": 15,
    "Sweet Peppers (Bell)": 16,
    "Tomato": 17,
    "Vegetables — Cucurbitaceae (general)": 18,
    "Cantaloupe": 19,
    "Pumpkin / Winter Squash": 22,
    "Squash / Zucchini": 23,
    "Sweet Melons": 24,
    "Watermelon": 25,
    "Roots & Tubers (general)": 26,
    "Beets (Table)": 27,
    "Parsnip": 30,
    "Potato": 31,
    "Sweet Potato": 32,
    "Turnip / Rutabaga": 33,
    "Sugar Beet": 34,
    "Legumes (general)": 35,
    "Chickpea": 38,
    "Garbanzo": 41,
    "Groundnut / Peanut": 44,
    "Lentil": 45,
    "Soybeans": 48,
    "Perennial Vegetables": 49,
    "Artichokes": 50,
    "Asparagus": 51,
    "Mint": 52,
    "Strawberries": 53,
    "Fiber Crops (general)": 54,
    "Cotton": 55,
    "Flax": 56,
    "Sisal": 57,
    "Oil Crops (general)": 58,
    "Castorbean": 59,
    "Rapeseed / Canola": 60,
    "Safflower": 61,
    "Sesame": 62,
    "Sunflower": 63,
    "Cereals (general)": 64,
    "Barley": 65,
    "Oats": 66,
    "Millet": 74,
    "Rice": 77,
    "Rye Grass (Hay)": 85,
    "Sugarcane": 92,
    "Cacao": 95,
    "Date Palms": 98,
    "Palm Trees": 99,
    "Rubber Trees": 102,
    "Berries & Bushes": 105,
    "Hops": 108,
    "Almonds": 109,
    "Avocado": 118,
    "Conifer Trees": 125,
    "Kiwi": 126,
    "Olives": 127,
    "Pistachios": 128,
    "Walnut": 129,
    "Short Vegetation": 132,
}

for _name, _idx in _SIMPLE.items():
    CROP_GROUPS[_name] = {"type": "simple", "index": _idx}

INDEX_TO_GROUP = {}
for _gname, _gdata in CROP_GROUPS.items():
    if _gdata["type"] == "ground_cover":
        for _i in _gdata["lookup"].values():
            INDEX_TO_GROUP[_i] = _gname
    elif _gdata["type"] == "variants":
        for _i in _gdata["options"].values():
            INDEX_TO_GROUP[_i] = _gname
    elif _gdata["type"] == "simple":
        INDEX_TO_GROUP[_gdata["index"]] = _gname


def get_table12_index(group_name, variant=None, ground_cover=None, secondary=None):
    gdata = CROP_GROUPS[group_name]
    if gdata["type"] == "simple":
        return gdata["index"]
    elif gdata["type"] == "variants":
        return gdata["options"][variant]
    elif gdata["type"] == "ground_cover":
        return gdata["lookup"][(ground_cover, secondary)]


def get_kc_values(idx):
    """Return (kc_ini, kc_mid, kc_end, hmax) from table12 row."""
    row = crops_df.iloc[idx]
    vals = []
    for col in ["Kcmini", "Kcmmid", "Kcmend", "hmax"]:
        try:
            vals.append(float(row[col]))
        except (ValueError, TypeError):
            vals.append(None)
    return vals


def adjust_kc(kc_ini, kc_mid, kc_end, hmax):
    """Apply FAO 56 Eq. 62 climate adjustment for Station 44."""
    if hmax is None or hmax <= 0:
        return kc_ini, kc_mid, kc_end
    adj = (0.04 * (MEAN_U2 - 2) - 0.004 * (MEAN_RH_MIN - 45)) * (hmax / 3) ** 0.3
    kc_mid_adj = (kc_mid + adj) if (kc_mid is not None and kc_mid > 0.45) else kc_mid
    kc_end_adj = (kc_end + adj) if (kc_end is not None and kc_end > 0.45) else kc_end
    return kc_ini, kc_mid_adj, kc_end_adj


# ============================================================
# Data Fetching
# ============================================================

CIMIS_API_KEY = st.secrets["CIMIS_API_KEY"]
STATION_ID = "44"

FIELD_MAP = {
    "DaySolRadAvg": "Sol Rad (W/sq.m)",
    "DayVapPresAvg": "Avg Vap Pres (kPa)",
    "DayAirTmpMax": "Max Air Temp (C)",
    "DayAirTmpMin": "Min Air Temp (C)",
    "DayAirTmpAvg": "Avg Air Temp (C)",
    "DayRelHumMax": "Max Rel Hum (%)",
    "DayRelHumMin": "Min Rel Hum (%)",
    "DayRelHumAvg": "Avg Rel Hum (%)",
    "DayDewPnt": "Dew Point (C)",
    "DayWindSpdAvg": "Avg Wind Speed (m/s)",
    "DaySoilTmpAvg": "Avg Soil Temp (C)",
    "DayEto": "ETo (mm)",
    "DayPrecip": "Precip (mm)",
}


@st.cache_data(ttl=3600)
def fetch_cimis_data():
    end = date.today()
    start = end - timedelta(days=20)
    params = {
        "appKey": CIMIS_API_KEY,
        "targets": STATION_ID,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "unitOfMeasure": "M",
        "dataItems": "day-eto,day-precip,day-sol-rad-avg,day-vap-pres-avg,"
        "day-air-tmp-max,day-air-tmp-min,day-air-tmp-avg,"
        "day-rel-hum-max,day-rel-hum-min,day-rel-hum-avg,"
        "day-dew-pnt,day-wind-spd-avg,day-soil-tmp-avg",
    }
    r = requests.get("https://et.water.ca.gov/api/data", params=params)
    records = r.json()["Data"]["Providers"][0]["Records"]

    rows = []
    for rec in records:
        row = {"Date": rec["Date"]}
        for api_field, col_name in FIELD_MAP.items():
            val = rec.get(api_field, {}).get("Value", None)
            row[col_name] = float(val) if val is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.ffill().bfill()
    df["Day of Year"] = df.index.dayofyear
    return df


@st.cache_data(ttl=3600)
def fetch_precip_forecast():
    """Fetch tomorrow's precipitation from NWS API."""
    headers = {"User-Agent": "(EToForecast App, student@cpp.edu)"}
    lat, lon = 33.9631, -117.3431

    try:
        r = requests.get(
            "https://api.weather.gov/points/%.4f,%.4f" % (lat, lon),
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        props = r.json()["properties"]
        forecast_url = props["forecast"]
        grid_url = "https://api.weather.gov/gridpoints/%s/%d,%d" % (
            props["gridId"],
            props["gridX"],
            props["gridY"],
        )

        # Text forecast
        r = requests.get(forecast_url, headers=headers, timeout=10)
        r.raise_for_status()
        periods = r.json()["properties"]["periods"]
        tomorrow = date.today() + timedelta(days=1)
        forecast_text = ""
        for period in periods:
            period_date = datetime.fromisoformat(period["startTime"]).date()
            if period_date == tomorrow and period["isDaytime"]:
                forecast_text = period.get("shortForecast", "")
                break

        # Quantitative precipitation
        r = requests.get(grid_url, headers=headers, timeout=10)
        r.raise_for_status()
        qpf_values = r.json()["properties"]["quantitativePrecipitation"]["values"]

        total_mm = 0.0
        for entry in qpf_values:
            time_part, duration_part = entry["validTime"].split("/")
            start = datetime.fromisoformat(time_part)
            match = re.match(r"PT(\d+)H", duration_part)
            duration_h = int(match.group(1)) if match else 0
            end = start + timedelta(hours=duration_h)
            local_start = (start - timedelta(hours=8)).date()
            local_end = (end - timedelta(hours=8)).date()
            if local_start == tomorrow or local_end == tomorrow:
                total_mm += entry.get("value", 0) or 0

        return {"precip_mm": round(total_mm, 1), "forecast": forecast_text}
    except Exception:
        return None


# ============================================================
# Prediction Functions
# ============================================================


def _build_input_tensor(df, start_idx, seq_len):
    """Scale a window of data and return input tensor."""
    input_cols = config["input_cols"]
    scale_cols = config["scale_cols"]
    window = df.iloc[start_idx : start_idx + seq_len]
    scaled = pd.DataFrame(scaler.transform(window[scale_cols]), columns=scale_cols)
    x = scaled[input_cols].values
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def _inverse_transform_eto(pred_scaled):
    """Convert a scaled ETo prediction back to mm."""
    scale_cols = config["scale_cols"]
    target_col = config["target_col"]
    dummy = np.zeros((1, len(scale_cols)))
    eto_idx = scale_cols.index(target_col)
    dummy[0, eto_idx] = pred_scaled
    return scaler.inverse_transform(dummy)[0, eto_idx]


def predict_tomorrow(df):
    """Predict tomorrow's ETo and return prediction + attention weights."""
    seq_len = config["seq_len"]
    x = _build_input_tensor(df, len(df) - seq_len, seq_len)

    with torch.no_grad():
        pred_scaled, attn_weights = model.predict_with_attention(x)

    pred_mm = _inverse_transform_eto(pred_scaled.item())
    weights = attn_weights.squeeze().numpy()
    dates = df.index[-seq_len:]

    return pred_mm, weights, dates


def backtest_recent(df):
    """Run model on recent windows and compare to actual ETo."""
    seq_len = config["seq_len"]
    results = []

    for i in range(seq_len, len(df)):
        x = _build_input_tensor(df, i - seq_len, seq_len)
        with torch.no_grad():
            pred_scaled = model(x).item()
        pred_mm = _inverse_transform_eto(pred_scaled)
        actual_mm = df.iloc[i]["ETo (mm)"]

        results.append(
            {
                "Date": df.index[i],
                "Predicted": round(pred_mm, 2),
                "Actual": round(actual_mm, 2),
            }
        )

    return pd.DataFrame(results).set_index("Date")


# ============================================================
# Sidebar — All Inputs
# ============================================================

# --- Crop ---
with st.sidebar.container(border=True):
    st.markdown("**Crop**")

    query = st.text_input(
        "Search crop", placeholder="e.g. orange, corn, turf, avocado..."
    )

    if query.strip():
        q = query.strip().lower()
        try:
            results = fao_tables.search12(query.strip())
            if not results.empty:
                filtered = []
                for i in results.index:
                    if i not in INDEX_TO_GROUP:
                        continue
                    group = INDEX_TO_GROUP[i]
                    crop_name = crops_df.iloc[i]["Crop"].lower()
                    if q in crop_name and q not in group.lower():
                        # Query matched a variant description, not the crop
                        continue
                    filtered.append(group)
                matching_groups = list(dict.fromkeys(filtered))
            else:
                matching_groups = []
        except Exception:
            matching_groups = []
    else:
        matching_groups = sorted(CROP_GROUPS.keys())

    kc = None

    if matching_groups:
        default_idx = (
            matching_groups.index("Citrus") if "Citrus" in matching_groups else 0
        )
        selected_group = st.selectbox(
            "Crop", matching_groups, index=default_idx, label_visibility="collapsed"
        )
        gdata = CROP_GROUPS[selected_group]

        if gdata["type"] == "ground_cover":
            gc = st.radio("Ground cover", ["None", "Active"], horizontal=True)
            secondary = st.radio(
                gdata["secondary_label"], gdata["secondary_options"], horizontal=True
            )
            idx = get_table12_index(
                selected_group, ground_cover=gc, secondary=secondary
            )
        elif gdata["type"] == "variants":
            variant_names = list(gdata["options"].keys())
            if len(variant_names) == 1:
                selected_variant = variant_names[0]
            else:
                selected_variant = st.selectbox("Variant", variant_names)
            idx = get_table12_index(selected_group, variant=selected_variant)
        else:
            idx = get_table12_index(selected_group)

        kc_ini, kc_mid, kc_end, hmax = get_kc_values(idx)
        kc_ini, kc_mid, kc_end = adjust_kc(kc_ini, kc_mid, kc_end, hmax)

        all_exist = all(v is not None for v in [kc_ini, kc_mid, kc_end])
        all_equal = all_exist and (
            abs(kc_ini - kc_mid) < 0.001 and abs(kc_mid - kc_end) < 0.001
        )

        if all_equal:
            kc = kc_mid
            st.markdown("**Kc = %.2f** (year-round, climate-adjusted)" % kc)
        elif all_exist:
            stage = st.selectbox(
                "Growth stage", ["Initial", "Mid-season", "Late season"]
            )
            kc = {"Initial": kc_ini, "Mid-season": kc_mid, "Late season": kc_end}[stage]
            st.markdown("**Kc = %.2f** (%s, climate-adjusted)" % (kc, stage.lower()))
        else:
            available = {}
            if kc_ini is not None:
                available["Initial"] = kc_ini
            if kc_mid is not None:
                available["Mid-season"] = kc_mid
            if kc_end is not None:
                available["Late season"] = kc_end
            if available:
                stage = st.selectbox("Growth stage", list(available.keys()))
                kc = available[stage]
                st.markdown(
                    "**Kc = %.2f** (%s, climate-adjusted)" % (kc, stage.lower())
                )

    else:
        st.caption("No matching crops found.")

    if kc is None:
        kc = st.number_input(
            "Custom Kc", min_value=0.0, max_value=2.0, value=0.65, step=0.05
        )

# --- Tomorrow's Weather ---
with st.sidebar.container(border=True):
    st.markdown("**Tomorrow's Weather**")

    precip_data = fetch_precip_forecast()

    if precip_data is not None and precip_data["forecast"]:
        st.caption(
            "NWS: %s  \n"
            "[National Weather Service](https://forecast.weather.gov)"
            % precip_data["forecast"]
        )

    default_precip = float(precip_data["precip_mm"]) if precip_data else 0.0
    tomorrow_precip = st.number_input(
        "Expected precipitation (mm)",
        min_value=0.0,
        max_value=200.0,
        value=default_precip,
        step=0.5,
        help=(
            "Pre-filled from NWS forecast. Adjust if needed."
            if precip_data
            else "NWS unavailable. Enter expected rainfall."
        ),
    )

# --- Irrigation ---
with st.sidebar.container(border=True):
    st.markdown("**Irrigation**")

    efficiency_options = {
        "Drip (90%)": 0.90,
        "Sprinkler (75%)": 0.75,
        "Flood / Furrow (60%)": 0.60,
        "Custom": None,
    }
    efficiency_choice = st.selectbox("Method", list(efficiency_options.keys()))
    if efficiency_choice == "Custom":
        efficiency = (
            st.number_input(
                "Efficiency (%)", min_value=10, max_value=100, value=80, step=5
            )
            / 100.0
        )
    else:
        efficiency = efficiency_options[efficiency_choice]


# ============================================================
# Fetch Data and Run Prediction
# ============================================================

with st.spinner("Fetching weather data from CIMIS..."):
    df = fetch_cimis_data()

predicted_eto, attn_weights, attn_dates = predict_tomorrow(df)
predicted_etc = kc * predicted_eto
water_needed = max(predicted_etc - tomorrow_precip, 0)
irrigation_needed = water_needed / efficiency if efficiency > 0 else water_needed


# ============================================================
# Main Page
# ============================================================

st.title("ETo Forecast")
st.caption("CIMIS Station 44 · UC Riverside")

# Hero — the actionable answer, always visible
with st.container(border=True):
    st.markdown(
        "<p style='margin:0; font-weight:600;'>Tomorrow's Irrigation</p>"
        "<p style='margin:0; font-size:2.5rem; font-weight:700;'>%.2f mm</p>"
        % irrigation_needed,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Apply %.1f L/m² of water tomorrow "
        "to meet crop demand, accounting for %s efficiency."
        % (irrigation_needed, efficiency_choice.lower())
    )

tab_forecast, tab_model = st.tabs(["Forecast", "Model"])


# --- Tab 1: Forecast ---
with tab_forecast:

    # Secondary metrics — the components
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted ETo", "%.2f mm" % predicted_eto)
    col2.metric("Crop ET (ETc)", "%.2f mm" % predicted_etc)
    col3.metric(
        "Expected Precip",
        "%.1f mm" % tomorrow_precip,
        help="Set in sidebar under Tomorrow's Weather. "
        "Default value from NWS forecast (api.weather.gov).",
    )

    # Formula breakdown — readable, line by line
    with st.container(border=True):
        st.markdown("**Calculation**")
        st.code(
            "ETc  = Kc × ETo\n"
            "     = %.2f × %.2f\n"
            "     = %.2f mm\n"
            "\n"
            "Irrigation = (ETc − precip) ÷ efficiency\n"
            "           = (%.2f − %.1f) ÷ %.0f%%\n"
            "           = %.2f mm"
            % (
                kc,
                predicted_eto,
                predicted_etc,
                predicted_etc,
                tomorrow_precip,
                efficiency * 100,
                irrigation_needed,
            ),
            language=None,
        )

    st.divider()

    # Recent ETo chart — full width
    st.subheader("Recent ETo")
    st.markdown(
        "Observed ETo at Station 44 over the last 14 days. "
        "The model uses this window of weather data to predict tomorrow's value."
    )
    recent_eto = df[["ETo (mm)"]].iloc[-14:].reset_index()
    chart = (
        alt.Chart(recent_eto)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("ETo (mm):Q", title="ETo (mm)"),
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Raw data
    with st.expander("View raw weather data"):
        display_cols = [
            "Max Air Temp (C)",
            "Min Air Temp (C)",
            "Avg Rel Hum (%)",
            "Sol Rad (W/sq.m)",
            "Avg Wind Speed (m/s)",
            "ETo (mm)",
            "Precip (mm)",
        ]
        st.dataframe(df[display_cols].iloc[-14:].style.format("{:.1f}"))


# --- Tab 2: Model ---
with tab_model:

    # 1. Prediction Pipeline
    st.subheader("Prediction Pipeline")
    st.markdown(
        "The model takes 14 days of weather observations (temperature, humidity, "
        "solar radiation, wind speed, and 8 other features) and predicts tomorrow's "
        "reference evapotranspiration (ETo). A 2-layer LSTM reads the sequence to "
        "capture temporal patterns, then a learned attention mechanism weights each "
        "day by its importance to the prediction. The weighted summary passes through "
        "fully connected layers to produce a single ETo value in mm/day."
    )

    st.markdown(
        "<style>"
        ".arrow-h, .arrow-v {font-size:1.5rem; text-align:center; padding-top:1rem;}"
        ".arrow-v {display:none;}"
        "@media (max-width:640px) {.arrow-h {display:none;} .arrow-v {display:block;}}"
        "</style>",
        unsafe_allow_html=True,
    )

    cols = st.columns([4, 1, 4, 1, 4, 1, 4, 1, 4])
    cols[0].container(border=True).markdown("**Weather Data**  \n14 days × 12 features")
    cols[1].markdown(
        "<p class='arrow-h'>→</p><p class='arrow-v'>↓</p>", unsafe_allow_html=True
    )
    cols[2].container(border=True).markdown("**LSTM**  \n2-layer, 128 hidden")
    cols[3].markdown(
        "<p class='arrow-h'>→</p><p class='arrow-v'>↓</p>", unsafe_allow_html=True
    )
    cols[4].container(border=True).markdown("**Attention**  \nLearned weights")
    cols[5].markdown(
        "<p class='arrow-h'>→</p><p class='arrow-v'>↓</p>", unsafe_allow_html=True
    )
    cols[6].container(border=True).markdown("**FC Layers**  \n128 → 32 → 1")
    cols[7].markdown(
        "<p class='arrow-h'>→</p><p class='arrow-v'>↓</p>", unsafe_allow_html=True
    )
    cols[8].container(border=True).markdown("**Output**  \nETo (mm/day)")

    st.divider()

    # 2. Attention Weights
    st.subheader("Attention Weights")
    st.markdown(
        "The attention weights for the current prediction. Each bar shows how much "
        "influence that day had on tomorrow's forecast."
    )

    attn_df = pd.DataFrame(
        {
            "Date": attn_dates.strftime("%b %d"),
            "Weight": attn_weights,
        }
    )
    attn_chart = (
        alt.Chart(attn_df)
        .mark_bar()
        .encode(
            x=alt.X("Date:N", sort=None, title=None),
            y=alt.Y("Weight:Q", title="Attention Weight"),
            color=alt.Color("Weight:Q", scale=alt.Scale(scheme="blues"), legend=None),
        )
        .properties(height=250)
    )
    st.altair_chart(attn_chart, use_container_width=True)

    # Weather context table
    with st.expander("Weather context for each day"):
        seq_len = config["seq_len"]
        context_df = df.iloc[-seq_len:][
            [
                "Max Air Temp (C)",
                "Sol Rad (W/sq.m)",
                "Avg Rel Hum (%)",
                "Avg Wind Speed (m/s)",
                "ETo (mm)",
            ]
        ].copy()
        context_df.insert(0, "Attention", [round(w, 3) for w in attn_weights])
        context_df.index = context_df.index.strftime("%b %d")
        context_df.index.name = "Date"
        context_df = context_df.sort_values("Attention", ascending=False)
        st.dataframe(
            context_df.style.format(
                {
                    "Attention": "{:.3f}",
                    "Max Air Temp (C)": "{:.1f}",
                    "Sol Rad (W/sq.m)": "{:.0f}",
                    "Avg Rel Hum (%)": "{:.0f}",
                    "Avg Wind Speed (m/s)": "{:.1f}",
                    "ETo (mm)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    st.divider()

    # 3. Recent Accuracy
    st.subheader("Recent Accuracy")
    st.markdown(
        "For each of the last several days, the model is run on the preceding "
        "14-day window from CIMIS and compared to the actual observed ETo."
    )

    backtest_df = backtest_recent(df)
    if not backtest_df.empty:
        st.line_chart(backtest_df, height=300)

        errors = (backtest_df["Predicted"] - backtest_df["Actual"]).abs()
        recent_mae = errors.mean()
        recent_rmse = np.sqrt((errors**2).mean())
        col1, col2, col3 = st.columns(3)
        col1.metric("Recent MAE", "%.2f mm" % recent_mae)
        col2.metric("Recent RMSE", "%.2f mm" % recent_rmse)
        col3.metric(
            "Test Set MAE", "0.636 mm", help="MAE on held-out test set (2024-2026)"
        )

    st.markdown(
        "The model achieves a test MAE of 0.636 mm/day on a mean ETo of "
        "4.1 mm/day, or about 15.5%% relative error. Published LSTM results "
        "on similar station data range from 0.30 to 0.70 mm/day "
        "([Roy et al. 2022](https://doi.org/10.3390/agronomy12030594), "
        "[Li et al. 2024](https://doi.org/10.1016/j.jhydrol.2024.132223)), "
        "with the best results using techniques like signal decomposition or "
        "remote sensing inputs that are outside the scope of this project."
    )
    st.markdown(
        "For practical irrigation use, the 0.636 mm ETo error translates to "
        "roughly 0.4 mm in the final recommendation after applying the crop "
        "coefficient. This is smaller than the typical distribution variability "
        "of drip or sprinkler systems (10-15%%), meaning the model is more "
        "precise than the hardware delivering the water. "
        "The largest errors occur on rain days and extreme high-ETo days, "
        "where atmospheric variability is hardest to predict from recent "
        "history alone."
    )

    st.divider()

    # 4. Architecture Details (expander)
    with st.expander("Architecture Details"):
        st.markdown(
            "**LSTM.** Weather is sequential. Yesterday's temperature, humidity, "
            "and radiation influence today's evapotranspiration. The LSTM's hidden "
            "state carries information forward through the 14-day window, allowing "
            "the model to learn patterns like multi-day heat buildups or post-rain "
            "recovery.\n\n"
            "**Attention.** A standard LSTM compresses the entire 14-day sequence "
            "into a single final hidden state. The attention mechanism lets the model "
            "look back at every day in the window and weight each one by its relevance "
            "to tomorrow's prediction. This also provides interpretability: the "
            "attention weights show which days the model considered most important.\n\n"
            "**L1Loss.** The combination of attention and L1Loss achieved the best "
            "result, though the individual contributions were not isolated. Attention "
            "with MSELoss performed worse than the baseline, while the same "
            "architecture with L1Loss performed best. Beyond accuracy, attention "
            "provides the interpretability shown in the weights chart above.\n\n"
            "**Hyperparameters.** The 14-day input window was informed by "
            "[Jia et al. (2023)](https://doi.org/10.1371/journal.pone.0281478), "
            "who found an optimal lookback of 22 days using hyperparameter "
            "search. 128 hidden units provide enough capacity without overfitting on "
            "the ~8,000 training sequences."
        )
        st.markdown(
            "| Parameter | Value |\n"
            "| :--- | :--- |\n"
            "| Architecture | LSTM + Attention |\n"
            "| Input | 14 days x 12 weather features |\n"
            "| Hidden size | 128 |\n"
            "| Layers | 2 |\n"
            "| Loss function | L1Loss (MAE) |\n"
            "| Optimizer | Adam |\n"
            "| Training data | 2000-2021 (8,022 sequences) |\n"
            "| Validation data | 2022-2023 |\n"
            "| Test MAE | 0.636 mm/day |\n"
            "| Test RMSE | 0.932 mm/day |"
        )

    # 5. Design Decisions (expander)
    with st.expander("Design Decisions"):
        st.markdown(
            "Seven model variants were evaluated under the same train/val/test protocol. "
            "The most significant finding: adding attention with MSELoss actually worsened "
            "performance (MAE 0.702 vs 0.685 baseline). Switching to L1Loss resolved this. "
            "MSE's quadratic penalty pushes the model toward safe, mean-centered predictions, "
            "giving attention weights weak gradients to learn from. L1Loss penalizes all errors "
            "equally, letting the model predict confidently on high-ETo days. The combination "
            "of attention + L1Loss achieved the best result: MAE 0.636 mm/day, a 7.2%% "
            "improvement over baseline.\n\n"
            "| Variant | MAE (mm/day) | RMSE (mm/day) |\n"
            "| :--- | :--- | :--- |\n"
            "| Baseline LSTM (MSELoss) | 0.685 | 0.934 |\n"
            "| LSTM + Attention (MSELoss) | 0.702 | 0.952 |\n"
            "| **LSTM + Attention (L1Loss)** | **0.636** | **0.932** |\n\n"
            "Other approaches evaluated include bidirectional LSTM, CNN-LSTM hybrid, "
            "and adding past ETo as an input feature. None improved on the baseline "
            "for this dataset."
        )


# About
st.markdown("---")
st.markdown("### About")
st.markdown(
    "Reference evapotranspiration (ETo) is the key input to irrigation planning, "
    "but it can only be calculated from observed weather. It cannot be forecasted. "
    "Irrigators need tomorrow's ETo today to schedule watering in advance. "
    "This app uses an LSTM neural network trained on 22 years of CIMIS weather data "
    "to predict next-day ETo, then converts that prediction into an irrigation "
    "recommendation using the FAO 56 crop water requirement formula."
)
st.markdown(
    "Lindsay Kislingbury · CS 4210 Machine Learning and Its Applications · Cal Poly Pomona  \n"
    "  \n"
    "Weather data: [CIMIS](https://cimis.water.ca.gov) Station 44, UC Riverside  \n"
    "Crop coefficients: FAO Irrigation and Drainage Paper 56 via "
    "[pyfao56](https://pypi.org/project/pyfao56/)  \n"
    "Precipitation forecast: [National Weather Service](https://weather.gov) API"
)
