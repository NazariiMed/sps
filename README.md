# SPS Sintering Data Smoothing and Regression

This project provides tools for analyzing and predicting the "Relative Piston Travel" in Spark Plasma Sintering (SPS) processes. It includes scripts for data exploration, smoothing, and regression modeling.

## Problem Overview

The original dataset contains "Rel. Piston Trav" values with limited precision, which results in numerous plateaus in the data (consecutive identical values). This lack of precision can hinder the performance of regression models. The smoothing scripts address this issue by adding controlled noise to break the plateaus while preserving the overall trends.

## Files and Scripts

### Original Scripts
- `data-exploration.py`: Performs exploratory data analysis and visualization
- `sintering-regression.py`: Implements three regression approaches for predicting "Rel. Piston Trav"

### New Scripts
- `smooth_data.py`: Creates smoothed versions of the original CSV files
- `verify_smoothing.py`: Verifies the quality of the smoothing and visualizes the improvements
- `use_smoothed_data.py`: Helper script to switch between original and smoothed data in the regression pipeline

## How to Use

### 1. Creating Smoothed Data

Run the `smooth_data.py` script to create smoothed versions of the original CSV files:

```bash
python smooth_data.py
```

This will:
- Load each of the original CSV files
- Analyze the "Rel. Piston Trav" column to determine appropriate smoothing parameters
- Apply a small amount of controlled noise to break plateaus
- Generate visualizations comparing original and smoothed data
- Save new CSV files with "_smoothed" suffix

### 2. Verifying the Smoothing

After creating the smoothed files, run the verification script to ensure the smoothing was effective:

```bash
python verify_smoothing.py
```

This will:
- Compare original and smoothed files
- Analyze the reduction in consecutive repeated values
- Generate various visualizations showing the improvements
- Confirm that the overall data distribution is preserved

### 3. Running Regression with Smoothed Data

You can use the `use_smoothed_data.py` script to switch between original and smoothed data for regression:

```bash
# Switch to smoothed data and run regression with approach 2
python use_smoothed_data.py --smoothed --approach 2

# Switch back to original data
python use_smoothed_data.py --approach 2
```

Alternatively, you can manually edit the file paths in `sintering-regression.py` to use the smoothed files.

## Regression Approaches

The regression script (`sintering-regression.py`) implements three approaches:

1. **Standard Approach**: Predict "Rel. Piston Trav" based only on current parameter values
2. **Window Approach**: Use current and previous time step data (including previous "Rel. Piston Trav") for prediction
3. **Virtual Experiment**: Similar to the window approach, but using predicted values from previous steps instead of actual measurements

## Smoothing Methods

The `smooth_data.py` script supports three smoothing methods:

1. **Noise** (default): Adds small random noise to break plateaus
2. **Spline**: Uses spline interpolation for smoothing
3. **Rolling**: Uses rolling average for smoothing

The noise method is the default as it preserves the overall structure of the data while effectively breaking plateaus.

## Data Files

### Original Data
- `160508-1021-1000,0min,56kN.csv`
- `160508-1022-900,0min,56kN.csv`
- `200508-1023-1350,0min,56kN.csv`
- `200508-1024-1200,0min,56kN.csv`

### Smoothed Data (generated)
- `160508-1021-1000,0min,56kN_smoothed.csv`
- `160508-1022-900,0min,56kN_smoothed.csv`
- `200508-1023-1350,0min,56kN_smoothed.csv`
- `200508-1024-1200,0min,56kN_smoothed.csv`

## Expected Results

Using the smoothed data with the regression approaches should yield:

1. Improved model performance (higher R² scores, lower RMSE)
2. More stable predictions in the virtual experiment approach
3. Reduced plateau effects in the predictions
4. Better feature importance insights

## Requirements

The code requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- scipy (for spline smoothing)

You can install these dependencies using:
```bash
pip install -r requirements.txt
```