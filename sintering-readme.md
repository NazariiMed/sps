# SPS Sintering Regression Analysis

This code implements regression analysis for Spark Plasma Sintering (SPS) process data. It includes three different approaches to predicting the "Rel. Piston Trav." parameter based on other process parameters.

## Overview

The script provides a comprehensive framework for analyzing SPS sintering data using machine learning regression techniques. It implements three different prediction approaches:

1. **Standard Approach**: Predict the "Rel. Piston Trav." for each row independently based only on the current values of other parameters.
2. **Window Approach**: Predict the "Rel. Piston Trav." of row `n` using both the parameter values at row `n` and the parameter values (including "Rel. Piston Trav.") from row `n-1`.
3. **Virtual Experiment**: Similar to the window approach, but uses the predicted value of "Rel. Piston Trav." from the previous step instead of the actual value, enabling continuous prediction without relying on measured "Rel. Piston Trav." values.

## Requirements

The code requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- bayes_opt (for Bayesian optimization of hyperparameters)

You can install these dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm bayesian-optimization
```

## Data Files

The code expects the following CSV files:
- 200508102313500min56kN.csv
- 200508102412000min56kN.csv
- 160508102110000min56kN.csv
- 16050810229000min56kN.csv

These files should contain SPS sintering data with semicolon-separated values and European number format (comma as decimal separator).

## Configuration

The script includes several configuration options at the top of the file:

```python
# Configuration for regression approaches
APPROACH = 1  # 1: Standard approach, 2: Window approach, 3: Virtual experiment 
VALIDATION_FILE_INDEX = 3  # Use the 4th file for validation (0-indexed)
TARGET_COLUMN = 'Rel. Piston Trav'
EXCLUDED_COLUMNS = ['Abs. Piston Trav', 'Nr.', 'Datum', 'Zeit']  # Columns to exclude

# Feature selection (manual control)
# Set to None to use all available features
SELECTED_FEATURES = [
    'MTC1', 'MTC2', 'MTC3', 'Pyrometer', 'SV Temperature', 
    'SV Power', 'SV Force', 'AV Force', 'AV Rel. Pressure', 
    'I RMS', 'U RMS', 'Heating power'
]

# Model selection (set to True to include in the evaluation)
MODELS_TO_EVALUATE = {
    'Linear Regression': True,
    'Ridge': True,
    'Lasso': True,
    'ElasticNet': True,
    'Decision Tree': True,
    'Random Forest': True,
    'Gradient Boosting': True,
    'XGBoost': True,
    'LightGBM': True,
    'SVR': True,
    'KNN': True
}

# Hyperparameter tuning settings
TUNING_METHOD = 'bayesian'  # 'grid', 'random', 'bayesian'
CV_FOLDS = 5
N_ITER = 20  # Number of iterations for random/bayesian search
```

You can modify these settings to customize the analysis:

- `APPROACH`: Set to 1, 2, or 3 based on which approach you want to use
- `VALIDATION_FILE_INDEX`: Specify which file to use for validation (0-based index)
- `SELECTED_FEATURES`: Specify which features to include in the regression, or set to `None` to use all available features
- `MODELS_TO_EVALUATE`: Enable/disable specific regression models
- `TUNING_METHOD`: Choose the hyperparameter tuning method (grid search, random search, or Bayesian optimization)

## Usage

1. Place the CSV data files in the same directory as the script
2. Configure the settings as described above
3. Run the script:
   ```bash
   python sintering_regression.py
   ```

## Output

The script will output:
1. Console logs showing the progress and results of the regression analysis
2. Visualization plots:
   - Feature importance plots for the best model
   - Actual vs. predicted value plots
   - Time series prediction plots
   - Residual analysis plots

## Approaches in Detail

### 1. Standard Approach

This is the simplest approach where each row is treated independently. The regression model predicts the "Rel. Piston Trav." value based only on the current values of other parameters.

```
Input: [Parameters at time t]
Output: Predicted Rel. Piston Trav. at time t
```

### 2. Window Approach

In this approach, we use information from the previous time step to help predict the current "Rel. Piston Trav." value. The model uses both the current parameter values and the previous parameter values (including the previous "Rel. Piston Trav.").

```
Input: [Parameters at time t, Parameters at time t-1, Rel. Piston Trav. at time t-1]
Output: Predicted Rel. Piston Trav. at time t
```

### 3. Virtual Experiment

This approach builds on the Window Approach but enables continuous prediction without requiring real "Rel. Piston Trav." measurements after the initial value. Instead, it uses its own predictions from previous steps:

```
Initial input: [Parameters at time t=1, Parameters at time t=0, Known Rel. Piston Trav. at time t=0]
Output: Predicted Rel. Piston Trav. at time t=1

Next step input: [Parameters at time t=2, Parameters at time t=1, Predicted Rel. Piston Trav. at time t=1]
Output: Predicted Rel. Piston Trav. at time t=2

And so on...
```

This allows for a "virtual experiment" where you only need to provide the machine configuration parameters, and the model can predict how the "Rel. Piston Trav." will evolve throughout the sintering process.

## Extending the Code

The code is designed to be modular and extensible:

- To add new regression models, add them to the `models` and `param_grids` dictionaries in the `build_and_evaluate_models` function
- To add new preprocessing steps, modify the `preprocess_data` function
- To add new evaluation metrics, extend the `evaluate_model` function
- To create additional visualizations, add new plotting functions

## Improving Precision

As noted in your requirements, the "Rel. Piston Trav." values in the dataset have limited precision (2 decimal places). The code handles this by using float64 precision for all calculations, which ensures that small differences can be represented accurately in the model predictions, even if the original data had limited precision.
