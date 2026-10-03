import matplotlib
matplotlib.use("Agg")

import runpy


# -----------------------------------------
# STEP 13: Export Results
# -----------------------------------------

# Run main.py without opening chart windows
results = runpy.run_path("main.py")

# Get the results created by main.py
accuracy_table = results["accuracy_table"]
final_forecast = results["final_forecast"]


# Save the accuracy table
accuracy_table.to_csv(
    "backtest_results.csv",
    index=False
)

# Save the final forecast table
final_forecast.to_csv(
    "forecast_output.csv",
    index=False
)


print("\nResults exported successfully.")

print("\nCreated files:")
print("1. backtest_results.csv")
print("2. forecast_output.csv")