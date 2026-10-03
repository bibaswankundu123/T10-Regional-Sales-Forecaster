import pandas as pd
import numpy as np

file_path = "T10_Regional_sales_forecaster.xlsx"

# Check available sheets
excel_file = pd.ExcelFile(file_path)
print("Sheets:")
print(excel_file.sheet_names)

# Load the Monthly Sales sheet
df = pd.read_excel(file_path, sheet_name="Monthly Sales")

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nRegions:")
print(df["region"].unique())

print("\nRows per region:")
print(df.groupby("region").size())

print("\nDate range:")
print(df["month"].min(), "to", df["month"].max())

# Convert month to datetime
df["month"] = pd.to_datetime(df["month"])

# Sort data chronologically within each region
df = df.sort_values(["region", "month"]).reset_index(drop=True)

print("\nData after sorting:")
print(df.head())

# Check the first and last date for each region
print("\nDate range by region:")
print(
    df.groupby("region")["month"]
      .agg(["min", "max", "count"])
)

# Create train and test datasets
# First 30 months = training
# Last 6 months = testing

train_data = df.groupby("region", group_keys=False).head(30).copy()
test_data = df.groupby("region", group_keys=False).tail(6).copy()

print("\nTraining data shape:")
print(train_data.shape)

print("\nTesting data shape:")
print(test_data.shape)

print("\nTraining period:")
print(train_data["month"].min(), "to", train_data["month"].max())

print("\nTesting period:")
print(test_data["month"].min(), "to", test_data["month"].max())

print("\nTraining rows by region:")
print(train_data.groupby("region").size())

print("\nTesting rows by region:")
print(test_data.groupby("region").size())

# -----------------------------------------
# STEP 5: 3-Month Moving Average Forecast
# -----------------------------------------

def moving_average_forecast(train_series, forecast_periods, window=3):
    """
    Forecast future values using a moving average.
    """
    history = list(train_series)
    forecasts = []

    for _ in range(forecast_periods):
        forecast = sum(history[-window:]) / window
        forecasts.append(forecast)
        history.append(forecast)

    return forecasts


# Test the moving average model for each region
moving_average_results = []

for region in df["region"].unique():

    # Get training sales for this region
    region_train = train_data[
        train_data["region"] == region
    ].sort_values("month")

    # Generate 6 forecasts
    forecasts = moving_average_forecast(
        region_train["net_sales_inr"].values,
        forecast_periods=6,
        window=3
    )

    # Get actual test values
    region_test = test_data[
        test_data["region"] == region
    ].sort_values("month")

    # Store results
    for month, actual, forecast in zip(
        region_test["month"],
        region_test["net_sales_inr"],
        forecasts
    ):
        moving_average_results.append({
            "month": month,
            "region": region,
            "actual": actual,
            "forecast": forecast
        })


# Convert results into DataFrame
moving_average_results = pd.DataFrame(moving_average_results)

print("\nMoving Average Back-test Results:")
print(moving_average_results)

# -----------------------------------------
# STEP 6: Simple Exponential Smoothing
# -----------------------------------------

from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def exponential_smoothing_forecast(train_series, forecast_periods):
    """
    Forecast future values using Simple Exponential Smoothing.
    The model estimates the smoothing level from the training data.
    """
    model = SimpleExpSmoothing(
    train_series,
    initialization_method="estimated"
)
    fitted_model = model.fit(optimized=True)

    forecasts = fitted_model.forecast(forecast_periods)

    return forecasts.tolist()


# Test Exponential Smoothing for each region
exponential_smoothing_results = []

for region in df["region"].unique():

    # Get training sales for this region
    region_train = train_data[
        train_data["region"] == region
    ].sort_values("month")

    # Generate 6 forecasts
    forecasts = exponential_smoothing_forecast(
        region_train["net_sales_inr"].values,
        forecast_periods=6
    )

    # Get actual test values
    region_test = test_data[
        test_data["region"] == region
    ].sort_values("month")

    # Store results
    for month, actual, forecast in zip(
        region_test["month"],
        region_test["net_sales_inr"],
        forecasts
    ):
        exponential_smoothing_results.append({
            "month": month,
            "region": region,
            "actual": actual,
            "forecast": forecast
        })


# Convert results into DataFrame
exponential_smoothing_results = pd.DataFrame(
    exponential_smoothing_results
)

print("\nExponential Smoothing Back-test Results:")
print(exponential_smoothing_results)

# -----------------------------------------
# STEP 7: Simple Linear Regression Forecast
# -----------------------------------------

from sklearn.linear_model import LinearRegression


def regression_forecast(train_series, forecast_periods):
    """
    Forecast future sales using simple linear regression
    with time index as the only predictor.
    """

    # Create time index for training data
    X_train = np.arange(1, len(train_series) + 1).reshape(-1, 1)

    # Sales values
    y_train = np.array(train_series)

    # Create and train regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create future time indices
    X_future = np.arange(
        len(train_series) + 1,
        len(train_series) + forecast_periods + 1
    ).reshape(-1, 1)

    # Generate forecasts
    forecasts = model.predict(X_future)

    return forecasts.tolist()


# Test Linear Regression for each region
regression_results = []

for region in df["region"].unique():

    # Get training sales for this region
    region_train = train_data[
        train_data["region"] == region
    ].sort_values("month")

    # Generate 6 forecasts
    forecasts = regression_forecast(
        region_train["net_sales_inr"].values,
        forecast_periods=6
    )

    # Get actual test values
    region_test = test_data[
        test_data["region"] == region
    ].sort_values("month")

    # Store results
    for month, actual, forecast in zip(
        region_test["month"],
        region_test["net_sales_inr"],
        forecasts
    ):
        regression_results.append({
            "month": month,
            "region": region,
            "actual": actual,
            "forecast": forecast
        })


# Convert results into DataFrame
regression_results = pd.DataFrame(regression_results)

print("\nLinear Regression Back-test Results:")
print(regression_results)

# -----------------------------------------
# STEP 8: Calculate MAPE
# -----------------------------------------

def calculate_mape(actual, forecast):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Ignores zero actual values to avoid division by zero.
    """
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    # Keep only observations where actual sales are not zero
    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - forecast[mask]) / actual[mask]
        )
    ) * 100


# Calculate MAPE for each region and method
mape_results = []

for region in df["region"].unique():

    # Moving Average
    ma_region = moving_average_results[
        moving_average_results["region"] == region
    ]

    ma_mape = calculate_mape(
        ma_region["actual"],
        ma_region["forecast"]
    )

    # Exponential Smoothing
    es_region = exponential_smoothing_results[
        exponential_smoothing_results["region"] == region
    ]

    es_mape = calculate_mape(
        es_region["actual"],
        es_region["forecast"]
    )

    # Linear Regression
    reg_region = regression_results[
        regression_results["region"] == region
    ]

    reg_mape = calculate_mape(
        reg_region["actual"],
        reg_region["forecast"]
    )

    # Store results
    mape_results.append({
        "region": region,
        "Moving Average": ma_mape,
        "Exponential Smoothing": es_mape,
        "Linear Regression": reg_mape
    })


# Convert to DataFrame
mape_results = pd.DataFrame(mape_results)

print("\nMAPE Comparison (%):")
print(mape_results.to_string(index=False))

# -----------------------------------------
# STEP 9: Overall MAPE and Model Selection
# -----------------------------------------

# Calculate average MAPE across all regions
overall_mape = mape_results[
    ["Moving Average", "Exponential Smoothing", "Linear Regression"]
].mean()

print("\nOverall Average MAPE (%):")
print(overall_mape)

# Find the method with the lowest overall MAPE
best_model = overall_mape.idxmin()
best_mape = overall_mape.min()

print("\nSelected Model:")
print(best_model)

print("\nOverall MAPE of Selected Model:")
print(f"{best_mape:.2f}%")

# -----------------------------------------
# STEP 10: Final 6-Month Forecast
# -----------------------------------------

final_forecast_results = []

# Dynamically determine the next 6 forecast months
last_month = df["month"].max()

forecast_start_date = last_month + pd.DateOffset(months=1)

future_months = pd.date_range(
    start=forecast_start_date,
    periods=6,
    freq="MS"
)

for region in df["region"].unique():

    # Get all 36 months of known data for this region
    region_data = df[
        df["region"] == region
    ].sort_values("month")

  # Select the forecasting function based on the best model
    forecast_dispatch = {
        "Moving Average": lambda s, p: moving_average_forecast(
            s, p, window=3
        ),
        "Exponential Smoothing": exponential_smoothing_forecast,
        "Linear Regression": regression_forecast
    }

    chosen_forecast_func = forecast_dispatch[best_model]

    # Use the selected model with all 36 known months
    forecasts = chosen_forecast_func(
        region_data["net_sales_inr"].values,
        forecast_periods=6
    )

    # Store the 6 future forecasts
    for month, forecast in zip(
        future_months,
        forecasts
    ):
        final_forecast_results.append({
            "month": month,
            "region": region,
            "forecast_method": best_model,
            "forecast_sales_inr": forecast
        })


# Convert to DataFrame
final_forecast = pd.DataFrame(final_forecast_results)

print("\nFinal 6-Month Forecast:")
print(final_forecast.to_string(index=False))

# -----------------------------------------
# STEP 11: Forecast Charts
# -----------------------------------------

import matplotlib.pyplot as plt

# Create charts folder
import os
os.makedirs("charts", exist_ok=True)


for region in df["region"].unique():

    # Historical data for this region
    region_history = df[
        df["region"] == region
    ].sort_values("month")

    # Forecast data for this region
    region_forecast = final_forecast[
        final_forecast["region"] == region
    ].sort_values("month")

    # Create a separate figure for each region
    plt.figure(figsize=(12, 6))

    # Plot historical sales
    plt.plot(
        region_history["month"],
        region_history["net_sales_inr"],
        marker="o",
        label="Historical Sales"
    )

    # Plot forecast sales
    plt.plot(
        region_forecast["month"],
        region_forecast["forecast_sales_inr"],
        marker="o",
        linestyle="--",
        label="Forecast Sales"
    )

    # Mark the beginning of the forecast period
    plt.axvline(
    forecast_start_date,
    linestyle=":",
    label="Forecast Starts"
  )

    plt.title(f"{region} Regional Sales Forecast")
    plt.xlabel("Month")
    plt.ylabel("Net Sales (INR)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save chart
    filename = f"charts/{region.lower()}_sales_forecast.png"
    plt.savefig(filename, dpi=300)

    plt.close()

    # -----------------------------------------
# STEP 12: Final Accuracy Table
# -----------------------------------------

accuracy_table = mape_results.copy()

# Add overall average row
overall_row = pd.DataFrame([{
    "region": "Overall Average",
    "Moving Average": accuracy_table["Moving Average"].mean(),
    "Exponential Smoothing": accuracy_table["Exponential Smoothing"].mean(),
    "Linear Regression": accuracy_table["Linear Regression"].mean()
}])

accuracy_table = pd.concat(
    [accuracy_table, overall_row],
    ignore_index=True
)

# Round MAPE values
accuracy_table[
    ["Moving Average", "Exponential Smoothing", "Linear Regression"]
] = accuracy_table[
    ["Moving Average", "Exponential Smoothing", "Linear Regression"]
].round(2)

print("\nFinal Accuracy Table:")
print(accuracy_table.to_string(index=False))