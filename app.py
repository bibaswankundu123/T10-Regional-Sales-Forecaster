import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------
# Regional Sales Forecaster Dashboard
# -----------------------------------------

st.set_page_config(
    page_title="Regional Sales Forecaster",
    page_icon="📈",
    layout="wide"
)


# -----------------------------------------
# Load Data
# -----------------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel(
        "T10_Regional_sales_forecaster.xlsx",
        sheet_name="Monthly Sales"
    )

    df["month"] = pd.to_datetime(df["month"])

    df = df.sort_values(
        ["region", "month"]
    ).reset_index(drop=True)

    return df


df = load_data()


# Load existing model results
backtest = pd.read_csv("backtest_results.csv")
forecast = pd.read_csv("forecast_output.csv")


# -----------------------------------------
# Title
# -----------------------------------------

st.title("📈 Regional Sales Forecaster")

st.write(
    "Six-month regional sales forecast based on "
    "historical monthly sales and back-tested forecasting models."
)


# -----------------------------------------
# Sidebar
# -----------------------------------------

st.sidebar.header("Dashboard Controls")

selected_region = st.sidebar.selectbox(
    "Select Region",
    sorted(df["region"].unique())
)


# -----------------------------------------
# KPI Section
# -----------------------------------------

region_forecast = forecast[
    forecast["region"] == selected_region
].copy()

total_forecast = region_forecast[
    "forecast_sales_inr"
].sum()

average_forecast = region_forecast[
    "forecast_sales_inr"
].mean()

selected_method = region_forecast[
    "forecast_method"
].iloc[0]


col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected Region",
    selected_region
)

col2.metric(
    "6-Month Forecast",
    f"₹{total_forecast:,.0f}"
)

col3.metric(
    "Forecast Method",
    selected_method
)


# -----------------------------------------
# Historical Sales Chart
# -----------------------------------------

st.subheader(
    f"{selected_region} — Historical & Forecast Sales"
)

region_history = df[
    df["region"] == selected_region
].sort_values("month")

region_forecast["month"] = pd.to_datetime(
    region_forecast["month"]
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    region_history["month"],
    region_history["net_sales_inr"],
    marker="o",
    label="Historical Sales"
)

ax.plot(
    region_forecast["month"],
    region_forecast["forecast_sales_inr"],
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

ax.set_xlabel("Month")
ax.set_ylabel("Net Sales (INR)")
ax.set_title(
    f"{selected_region} Sales Forecast"
)

ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# -----------------------------------------
# Forecast Table
# -----------------------------------------

st.subheader("Six-Month Forecast")

display_forecast = region_forecast[
    ["month", "forecast_method", "forecast_sales_inr"]
].copy()

display_forecast["month"] = display_forecast[
    "month"
].dt.strftime("%b-%Y")

display_forecast["forecast_sales_inr"] = (
    display_forecast["forecast_sales_inr"]
    .round(0)
)

display_forecast.columns = [
    "Month",
    "Forecast Method",
    "Forecast Sales (INR)"
]

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------------
# Model Accuracy
# -----------------------------------------

st.subheader("Model Accuracy Comparison")

accuracy_region = backtest[
    backtest["region"] == selected_region
].copy()

accuracy_region = accuracy_region.set_index(
    "region"
).T.reset_index()

accuracy_region.columns = [
    "Model",
    "MAPE (%)"
]

accuracy_region = accuracy_region[
    accuracy_region["Model"] != "region"
]

st.dataframe(
    accuracy_region,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------------
# Project Information
# -----------------------------------------

st.subheader("Project Information")

st.write(
    "Training period: April 2023 – September 2025"
)

st.write(
    "Back-test period: October 2025 – March 2026"
)

st.write(
    "Forecast period: April 2026 – September 2026"
)