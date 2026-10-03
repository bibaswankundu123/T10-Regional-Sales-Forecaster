# T10 – Regional Sales Forecaster

A regional sales forecasting project developed for the T10 mini-project using Python, Machine Learning/forecasting techniques, and Streamlit.

## Project Objective

The objective is to forecast the next 6 months of regional sales and evaluate different forecasting methods using historical monthly sales data.

The project covers four regions:
- East
- North
- South
- West

## Forecasting Methods

Three forecasting approaches were evaluated:

1. 3-Month Moving Average
2. Simple Exponential Smoothing
3. Simple Linear Regression

## Back-Testing

The dataset contains 36 months of monthly sales data for each region.

- Training period: First 30 months
- Back-test period: Last 6 months
- Evaluation metric: MAPE (Mean Absolute Percentage Error)

### Overall Back-Test Results

| Model | Overall MAPE |
|---|---:|
| Moving Average | 10.91% |
| Exponential Smoothing | 10.25% |
| Linear Regression | **8.77%** |

**Selected Model: Simple Linear Regression**

Linear Regression achieved the lowest overall MAPE and also produced the lowest MAPE for each of the four regions.

## Final Forecast

The selected model was refitted using all 36 known months of data and used to forecast:

**April 2026 – September 2026**

Forecasts are generated separately for East, North, South, and West.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides:

- Regional selection
- 6-month sales forecast
- Historical vs. forecast chart
- Forecast table
- Model accuracy comparison

## Project Files

- `main.py` – Forecasting and back-testing pipeline
- `app.py` – Streamlit dashboard
- `export_results.py` – Exports forecasting results
- `verify_results.py` – Validates exported results
- `backtest_results.csv` – Model accuracy results
- `forecast_output.csv` – Final 6-month forecasts
- `charts/` – Regional forecast charts
- `AI Evidence/` – Evidence of AI-assisted development
- `App Evidence/` – Dashboard screenshots
- `T10_Regional_sales_forecaster_completed.xlsx` – Completed project workbook
- `T10_Regional_Sales_Forecaster_Report.docx` – Project report

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
streamlit run app.py