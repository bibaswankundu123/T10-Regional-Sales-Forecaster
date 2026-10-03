import pandas as pd


# -----------------------------------------
# STEP 14: Verify Exported Results
# -----------------------------------------

# Load exported files
backtest = pd.read_csv("backtest_results.csv")
forecast = pd.read_csv("forecast_output.csv")


# Check Backtest Results
print("BACKTEST RESULTS")
print("-----------------")
print("Shape:", backtest.shape)
print(backtest.to_string(index=False))


# Check Forecast Output
print("\nFORECAST OUTPUT")
print("----------------")
print("Shape:", forecast.shape)
print(forecast.to_string(index=False))


# Basic validation checks
print("\nVALIDATION CHECKS")
print("-----------------")

print("Backtest rows should be 5:")
print(backtest.shape[0])

print("\nForecast rows should be 24:")
print(forecast.shape[0])

print("\nRegions in forecast:")
print(forecast["region"].unique())

print("\nForecast months:")
print(forecast["month"].unique())