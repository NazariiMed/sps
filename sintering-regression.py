import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths
file_paths = [
    '160508-1021-1000,0min,56kN.csv',
    '160508-1022-900,0min,56kN.csv',
    '200508-1023-1350,0min,56kN.csv',
    '200508-1024-1200,0min,56kN.csv'
]

# Configuration for regression approaches
APPROACH = 2  # 1: Standard approach, 2: Window approach, 3: Virtual experiment
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
    'LightGBM': False,
    'SVR': True,
    'KNN': True
}

# Hyperparameter tuning settings
TUNING_METHOD = 'bayesian'  # 'grid', 'random', 'bayesian'
CV_FOLDS = 5
N_ITER = 20  # Number of iterations for random/bayesian search


def load_data(file_paths, validation_index):
    """
    Load and preprocess the CSV files.

    Args:
        file_paths: List of CSV file paths
        validation_index: Index of the file to use for validation

    Returns:
        train_data: Combined DataFrame of training data
        validation_data: DataFrame for validation
    """
    all_data = []

    for i, file_path in enumerate(file_paths):
        # Read the CSV file with proper settings for European number format
        print(f"Loading file: {file_path}")
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',', header=0)
            print(f"  File shape: {df.shape}")

            # Add a file identifier column
            df['file_id'] = i

            all_data.append(df)
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    if not all_data:
        raise ValueError("No data files could be loaded!")

    # Split into training and validation
    validation_data = all_data.pop(validation_index)
    print(f"Validation data shape: {validation_data.shape}")

    train_data = pd.concat(all_data, ignore_index=True)
    print(f"Training data shape: {train_data.shape}")

    return train_data, validation_data


def preprocess_data(df, target_col, excluded_cols, selected_features=None):
    """
    Preprocess the data for regression.

    Args:
        df: Input DataFrame
        target_col: Target column name
        excluded_cols: List of columns to exclude
        selected_features: List of features to include (None = use all)

    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names used
    """
    # Make a copy to avoid modifying the original
    data = df.copy()

    # Check if target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {data.columns.tolist()}")

    print(f"Preprocessing data with shape: {data.shape}")
    print(f"Target column: {target_col}")

    # Drop rows with NaN in target column
    original_count = len(data)
    data = data.dropna(subset=[target_col])
    dropped_count = original_count - len(data)
    print(f"Dropped {dropped_count} rows with missing target values")

    # Extract target
    y = data[target_col].values

    # Convert -999 values to NaN (likely error codes in the dataset)
    data = data.replace(-999, np.nan)

    # Drop specified columns and the target
    columns_to_drop = excluded_cols + [target_col, 'file_id']
    X_data = data.drop(columns=columns_to_drop, errors='ignore')

    # Select only specified features if provided
    if selected_features is not None:
        available_features = [col for col in selected_features if col in X_data.columns]
        missing_features = [col for col in selected_features if col not in X_data.columns]
        if missing_features:
            print(f"Warning: Some selected features are not in the data: {missing_features}")
        X_data = X_data[available_features]

    print(f"Selected features: {X_data.columns.tolist()}")

    # Check for non-numeric columns
    non_numeric = X_data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Warning: Non-numeric columns found: {non_numeric}")
        print("Converting to numeric or dropping...")

        for col in non_numeric:
            try:
                # Try to convert to numeric
                X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
            except:
                # If conversion fails, drop the column
                print(f"  Dropping column: {col}")
                X_data = X_data.drop(columns=[col])

    # Check for NaN values
    nan_count = X_data.isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in features. Filling with column means...")

    # Fill remaining NaNs with column means
    X_data = X_data.fillna(X_data.mean())

    # Get feature names for later use
    feature_names = X_data.columns.tolist()

    # Convert to numpy array for modeling
    X = X_data.values

    # Improve precision of target variable (if needed)
    # This doesn't change the actual precision but makes sure we're using float64
    y = y.astype(np.float64)

    print(f"Preprocessed data: X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_names


def prepare_window_data(X, y, window_size=1):
    """
    Prepare data for window-based approach (Approach 2 & 3).

    Args:
        X: Feature matrix
        y: Target vector
        window_size: Number of previous steps to include

    Returns:
        X_window: Feature matrix with window features
        y_window: Target vector aligned with the window features
    """
    n_samples, n_features = X.shape

    # We need at least window_size+1 samples to create a valid window
    if n_samples <= window_size:
        raise ValueError(f"Not enough samples ({n_samples}) for window size {window_size}")

    # Initialize arrays for the windowed data
    X_window = np.zeros((n_samples - window_size, n_features * (window_size + 1) + window_size))
    y_window = np.zeros(n_samples - window_size)

    # Fill in the arrays
    for i in range(window_size, n_samples):
        # Current features
        X_window[i - window_size, :n_features] = X[i]

        # Add previous features and targets
        for w in range(1, window_size + 1):
            # Previous features
            start_idx = n_features + (w - 1) * n_features
            end_idx = start_idx + n_features
            X_window[i - window_size, start_idx:end_idx] = X[i - w]

            # Previous target
            X_window[i - window_size, n_features * (window_size + 1) + (w - 1)] = y[i - w]

        # Current target
        y_window[i - window_size] = y[i]

    return X_window, y_window


def virtual_experiment(model, X_val, y_val, feature_names, window_size=1):
    """
    Run a virtual experiment with the trained model (Approach 3).

    Args:
        model: Trained regression model
        X_val: Validation feature matrix
        y_val: Validation target vector
        feature_names: List of feature names
        window_size: Window size used in training

    Returns:
        y_pred: Predicted target values
        y_true: Actual target values
    """
    n_samples, n_features = X_val.shape
    n_features_per_window = int(X_val.shape[1] / (window_size + 1) - window_size)

    # We'll store actual and predicted values
    y_true = []
    y_pred = []

    # Initialize with the first few actual values
    # (in a real scenario, we might have initial measurements)
    prev_y_values = y_val[:window_size].tolist()

    # Process each time step
    for i in range(window_size, n_samples):
        # Extract current features (the machine settings for this step)
        current_features = X_val[i, :n_features_per_window]

        # Construct input for the model using current features and previous info
        model_input = np.zeros(n_features)

        # Add current features
        model_input[:n_features_per_window] = current_features

        # Add previous features and predicted targets
        for w in range(1, window_size + 1):
            # Previous features
            prev_features_idx = i - w
            start_idx = n_features_per_window + (w - 1) * n_features_per_window
            end_idx = start_idx + n_features_per_window

            # Use actual previous features from validation set
            model_input[start_idx:end_idx] = X_val[prev_features_idx, :n_features_per_window]

            # Use predicted previous target instead of actual value
            prev_y_idx = n_features_per_window * (window_size + 1) + (w - 1)
            model_input[prev_y_idx] = prev_y_values[-w]

        # Make prediction
        prediction = model.predict([model_input])[0]

        # Store actual and predicted values
        y_true.append(y_val[i])
        y_pred.append(prediction)

        # Update previous y values for next iteration
        prev_y_values.append(prediction)

    return np.array(y_pred), np.array(y_true)


def analyze_target_precision(df, target_col, plot=True):
    """
    Analyze the precision issues in the target column.
    
    Args:
        df: DataFrame containing the target column
        target_col: Name of the target column
        plot: Whether to generate plots
        
    Returns:
        dict: Dictionary with precision analysis results
    """
    print(f"\nAnalyzing precision issues in '{target_col}'...")
    
    # Extract target column
    target_values = df[target_col].values
    
    # Calculate differences between consecutive values
    differences = np.diff(target_values)
    non_zero_diffs = differences[differences != 0]
    
    # Ensure we have absolute differences for calculations that need positive values
    abs_non_zero_diffs = np.abs(non_zero_diffs)
    
    # Count occurrences of repeated values
    consecutive_repeats = []
    current_count = 1
    
    for i in range(1, len(target_values)):
        if abs(target_values[i] - target_values[i-1]) < 1e-10:
            current_count += 1
        else:
            if current_count > 1:
                consecutive_repeats.append(current_count)
            current_count = 1
    
    # Add the last group if it's a repeat
    if current_count > 1:
        consecutive_repeats.append(current_count)
    
    # Calculate statistics
    results = {
        'unique_values': df[target_col].nunique(),
        'total_values': len(target_values),
        'min_nonzero_diff': np.min(non_zero_diffs) if len(non_zero_diffs) > 0 else 0,
        'min_abs_nonzero_diff': np.min(abs_non_zero_diffs) if len(abs_non_zero_diffs) > 0 else 0.0001,
        'avg_nonzero_diff': np.mean(non_zero_diffs) if len(non_zero_diffs) > 0 else 0,
        'avg_abs_nonzero_diff': np.mean(abs_non_zero_diffs) if len(abs_non_zero_diffs) > 0 else 0.0001,
        'median_nonzero_diff': np.median(non_zero_diffs) if len(non_zero_diffs) > 0 else 0,
        'zero_diff_count': len(differences) - len(non_zero_diffs),
        'zero_diff_percentage': 100 * (len(differences) - len(non_zero_diffs)) / len(differences),
        'max_consecutive_repeats': max(consecutive_repeats) if consecutive_repeats else 0,
        'avg_consecutive_repeats': np.mean(consecutive_repeats) if consecutive_repeats else 0
    }
    
    # Print results
    print(f"  Unique values: {results['unique_values']} out of {results['total_values']} total values")
    print(f"  Minimum non-zero difference: {results['min_nonzero_diff']:.8f}")
    print(f"  Minimum absolute non-zero difference: {results['min_abs_nonzero_diff']:.8f}")
    print(f"  Average non-zero difference: {results['avg_nonzero_diff']:.8f}")
    print(f"  Average absolute non-zero difference: {results['avg_abs_nonzero_diff']:.8f}")
    print(f"  Zero differences: {results['zero_diff_count']} ({results['zero_diff_percentage']:.2f}% of all consecutive pairs)")
    print(f"  Maximum consecutive repeated values: {results['max_consecutive_repeats']}")
    print(f"  Average run length of repeated values: {results['avg_consecutive_repeats']:.2f}")
    
    # Generate plots if requested
    if plot:
        # Plot 1: Histogram of non-zero differences
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(non_zero_diffs, bins=50, alpha=0.7)
        plt.xlabel('Non-zero differences between consecutive values')
        plt.ylabel('Frequency')
        plt.title('Distribution of non-zero differences')
        
        # Plot 2: Time series of values
        plt.subplot(2, 2, 2)
        plt.plot(target_values[:1000])  # Just plot first 1000 for clarity
        plt.xlabel('Index')
        plt.ylabel(target_col)
        plt.title(f'{target_col} values (first 1000 points)')
        
        # Plot 3: Histogram of consecutive repeats
        plt.subplot(2, 2, 3)
        plt.hist(consecutive_repeats, bins=30, alpha=0.7)
        plt.xlabel('Number of consecutive repeats')
        plt.ylabel('Frequency')
        plt.title('Distribution of consecutive repeated values')
        
        # Plot 4: Original vs. smoothed data
        plt.subplot(2, 2, 4)
        
        # Sample points for demonstration
        sample = target_values[:1000]
        
        # Create a smoothed version by adding tiny noise
        noise_scale = results['min_abs_nonzero_diff']/10
        smoothed = sample + np.random.normal(0, noise_scale, len(sample))
        
        plt.plot(sample, label='Original', alpha=0.7)
        plt.plot(smoothed, label='With tiny noise', alpha=0.7)
        plt.xlabel('Index')
        plt.ylabel(target_col)
        plt.title('Original vs. noise-added values')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return results


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test target vector
        model_name: Name of the model for display

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print(f"  In evaluate_model - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return as dictionary
    metrics = {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    return metrics, y_pred


def scale_data(X_train, X_test):
    """
    Standardize features by removing mean and scaling to unit variance.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix

    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        scaler: The fitted scaler for future use
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def tune_hyperparameters(model_class, param_grid, X_train, y_train, method='grid', cv=5, n_iter=20):
    """
    Tune hyperparameters for a model.

    Args:
        model_class: Scikit-learn model class
        param_grid: Dictionary of hyperparameters
        X_train: Training feature matrix
        y_train: Training target vector
        method: Tuning method ('grid', 'random', or 'bayesian')
        cv: Number of cross-validation folds
        n_iter: Number of iterations for random/bayesian search

    Returns:
        best_model: Tuned model
        best_params: Best hyperparameter values
    """
    # Debug: Print shapes
    print(f"  In tune_hyperparameters - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    if method == 'grid':
        search = GridSearchCV(
            model_class(), param_grid, cv=cv, scoring='neg_mean_squared_error',
            verbose=0, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

    elif method == 'random':
        search = RandomizedSearchCV(
            model_class(), param_grid, n_iter=n_iter, cv=cv,
            scoring='neg_mean_squared_error', verbose=0, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

    elif method == 'bayesian':
        # For debugging, use RandomizedSearchCV instead of BayesSearchCV
        search = RandomizedSearchCV(
            model_class(), param_grid, n_iter=n_iter, cv=cv,
            scoring='neg_mean_squared_error', verbose=0, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

    else:
        raise ValueError(f"Unknown tuning method: {method}")

    # Debug: check fitted model
    print(f"  Fitted model: {best_model}")

    return best_model, best_params


def build_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Build and evaluate multiple regression models.

    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector

    Returns:
        results: List of evaluation metrics for each model
        best_model: The best performing model
        all_models: Dictionary of all trained models
    """
    # Print data dimensions for debugging
    print(f"Data dimensions before scaling:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Print data dimensions after scaling
    print(f"Data dimensions after scaling:")
    print(f"  X_train_scaled: {X_train_scaled.shape}")
    print(f"  X_test_scaled: {X_test_scaled.shape}")

    # Define models to evaluate
    models = {}
    param_grids = {}

    # Linear models
    if MODELS_TO_EVALUATE.get('Linear Regression', False):
        models['Linear Regression'] = LinearRegression()
        param_grids['Linear Regression'] = {'fit_intercept': [True, False]}

    if MODELS_TO_EVALUATE.get('Ridge', False):
        models['Ridge'] = Ridge()
        param_grids['Ridge'] = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }

    if MODELS_TO_EVALUATE.get('Lasso', False):
        models['Lasso'] = Lasso()
        param_grids['Lasso'] = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'max_iter': [1000, 3000, 5000]
        }

    if MODELS_TO_EVALUATE.get('ElasticNet', False):
        models['ElasticNet'] = ElasticNet()
        param_grids['ElasticNet'] = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'max_iter': [1000, 3000, 5000]
        }

    # Tree-based models
    if MODELS_TO_EVALUATE.get('Decision Tree', False):
        models['Decision Tree'] = DecisionTreeRegressor()
        param_grids['Decision Tree'] = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    if MODELS_TO_EVALUATE.get('Random Forest', False):
        models['Random Forest'] = RandomForestRegressor()
        param_grids['Random Forest'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    if MODELS_TO_EVALUATE.get('Gradient Boosting', False):
        models['Gradient Boosting'] = GradientBoostingRegressor()
        param_grids['Gradient Boosting'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10]
        }

    if MODELS_TO_EVALUATE.get('XGBoost', False):
        models['XGBoost'] = xgb.XGBRegressor()
        param_grids['XGBoost'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

    if MODELS_TO_EVALUATE.get('LightGBM', False):
        models['LightGBM'] = lgb.LGBMRegressor()
        param_grids['LightGBM'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0]
        }

    # Other models
    if MODELS_TO_EVALUATE.get('SVR', False):
        models['SVR'] = SVR()
        param_grids['SVR'] = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }

    if MODELS_TO_EVALUATE.get('KNN', False):
        models['KNN'] = KNeighborsRegressor()
        param_grids['KNN'] = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

    # Train and evaluate models
    results = []
    all_models = {}
    best_r2 = -float('inf')
    best_model = None
    best_model_name = None

    for name, model in models.items():
        print(f"Training {name}...")

        try:
            # Tune hyperparameters if grid is provided
            if name in param_grids and len(param_grids[name]) > 0:
                model_class = model.__class__
                tuned_model, best_params = tune_hyperparameters(
                    model_class, param_grids[name],
                    X_train_scaled, y_train,
                    method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER
                )
                print(f"  Best params: {best_params}")
                all_models[name] = tuned_model
            else:
                # Just fit the model with default parameters
                model.fit(X_train_scaled, y_train)
                all_models[name] = model

            # Evaluate model
            metrics, y_pred = evaluate_model(all_models[name], X_test_scaled, y_test, name)
            results.append(metrics)

            print(f"  {name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

            # Track best model
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = all_models[name]
                best_model_name = name

        except Exception as e:
            print(f"Error training {name}: {e}")

    print(f"\nBest model: {best_model_name} (R² = {best_r2:.4f})")

    # Sort results by R2 (higher is better)
    results.sort(key=lambda x: x['r2'], reverse=True)

    return results, best_model, all_models, scaler


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to plot
    """
    plt.figure(figsize=(12, 8))

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_)
    else:
        print(f"Model {model_name} does not have built-in feature importance.")
        return

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    # Plot top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.barh(range(len(top_indices)), top_importances, align='center')
    plt.yticks(range(len(top_indices)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, model_name, approach):
    """
    Plot actual vs predicted values.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name: Name of the model
        approach: Regression approach used
    """
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {model_name} (Approach {approach})')

    # Calculate metrics for the plot
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Add text with metrics
    plt.annotate(f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_time_series_prediction(y_true, y_pred, model_name, approach):
    """
    Plot time series of actual and predicted values.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name: Name of the model
        approach: Regression approach used
    """
    plt.figure(figsize=(15, 8))

    # Create time series plot
    plt.plot(range(len(y_true)), y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', alpha=0.7)

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Rel. Piston Trav')
    plt.title(f'Time Series Prediction - {model_name} (Approach {approach})')
    plt.legend()

    # Calculate metrics for the plot
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Add text with metrics
    plt.annotate(f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_residuals(y_true, y_pred, model_name, approach):
    """
    Plot residuals.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name: Name of the model
        approach: Regression approach used
    """
    residuals = y_true - y_pred

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    # Add overall title
    plt.suptitle(f'Residual Analysis - {model_name} (Approach {approach})', y=1.05)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_smoothing_comparison(y_true, y_pred_orig, y_pred_medium, y_pred_high, model_name):
    """
    Compare different smoothing approaches for the virtual experiment.
    
    Args:
        y_true: Actual target values
        y_pred_orig: Predicted values without smoothing
        y_pred_medium: Predicted values with medium smoothing
        y_pred_high: Predicted values with high smoothing
        model_name: Name of the model
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Calculate error metrics
    rmse_orig = np.sqrt(mean_squared_error(y_true, y_pred_orig))
    rmse_medium = np.sqrt(mean_squared_error(y_true, y_pred_medium))
    rmse_high = np.sqrt(mean_squared_error(y_true, y_pred_high))
    
    r2_orig = r2_score(y_true, y_pred_orig)
    r2_medium = r2_score(y_true, y_pred_medium)
    r2_high = r2_score(y_true, y_pred_high)
    
    # Plot 1: Full time series
    axes[0].plot(y_true, label='Actual', color='blue', alpha=0.7)
    axes[0].plot(y_pred_orig, label=f'No Smoothing (RMSE={rmse_orig:.4f}, R²={r2_orig:.4f})', 
             color='red', alpha=0.7)
    axes[0].plot(y_pred_medium, label=f'Medium Smoothing (RMSE={rmse_medium:.4f}, R²={r2_medium:.4f})', 
              color='green', alpha=0.7)
    axes[0].plot(y_pred_high, label=f'High Smoothing (RMSE={rmse_high:.4f}, R²={r2_high:.4f})', 
             color='purple', alpha=0.7)
    
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Rel. Piston Trav')
    axes[0].set_title('Full Time Series Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Identify plateau regions in original prediction
    plateaus = []
    consecutive_same = 0
    last_val = None
    
    for i, val in enumerate(y_pred_orig):
        if last_val is not None and abs(val - last_val) < 1e-6:
            consecutive_same += 1
            if consecutive_same >= 5:  # Consider it a plateau after 5 same values
                if len(plateaus) == 0 or plateaus[-1][1] < i - consecutive_same:
                    plateaus.append((i - consecutive_same, i))  # (start, end)
                else:
                    plateaus[-1] = (plateaus[-1][0], i)  # Extend the last plateau
        else:
            consecutive_same = 0
        last_val = val
    
    # Plot 2: Zoom in on a problematic region with plateaus
    if plateaus:
        # Find the longest plateau
        longest_plateau = max(plateaus, key=lambda x: x[1] - x[0])
        start_idx = max(0, longest_plateau[0] - 10)
        end_idx = min(len(y_true), longest_plateau[1] + 10)
        
        # Plot 2: Zoomed in on plateau region
        axes[1].plot(range(start_idx, end_idx), y_true[start_idx:end_idx], 
                 label='Actual', color='blue', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_orig[start_idx:end_idx], 
                 label='No Smoothing', color='red', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_medium[start_idx:end_idx], 
                  label='Medium Smoothing', color='green', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_high[start_idx:end_idx], 
                 label='High Smoothing', color='purple', alpha=0.7)
        
        # Highlight the plateau region
        plt_start, plt_end = longest_plateau
        axes[1].axvspan(plt_start, plt_end, alpha=0.2, color='yellow', label='Plateau')
        
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Rel. Piston Trav')
        axes[1].set_title(f'Zoomed View of Plateau Region (Steps {start_idx}-{end_idx})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # If no plateaus found, just show a zoomed portion of the middle
        mid = len(y_true) // 2
        start_idx = max(0, mid - 50)
        end_idx = min(len(y_true), mid + 50)
        
        axes[1].plot(range(start_idx, end_idx), y_true[start_idx:end_idx], 
                 label='Actual', color='blue', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_orig[start_idx:end_idx], 
                 label='No Smoothing', color='red', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_medium[start_idx:end_idx], 
                  label='Medium Smoothing', color='green', alpha=0.7)
        axes[1].plot(range(start_idx, end_idx), y_pred_high[start_idx:end_idx], 
                 label='High Smoothing', color='purple', alpha=0.7)
        
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Rel. Piston Trav')
        axes[1].set_title(f'Zoomed View (Steps {start_idx}-{end_idx})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Create a third plot showing the differences between smoothed and unsmoothed
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot differences between predictions
    diff_medium = y_pred_medium - y_pred_orig
    diff_high = y_pred_high - y_pred_orig
    
    axes2[0].plot(diff_medium, label='Medium Smoothing Diff', color='green', alpha=0.7)
    axes2[0].plot(diff_high, label='High Smoothing Diff', color='purple', alpha=0.7)
    axes2[0].axhline(y=0, color='r', linestyle='--')
    axes2[0].set_xlabel('Time Step')
    axes2[0].set_ylabel('Difference from Original')
    axes2[0].set_title('Difference Between Smoothed and Original Predictions')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Histogram of consecutive same values in the original prediction
    consecutive_counts = []
    count = 1
    for i in range(1, len(y_pred_orig)):
        if abs(y_pred_orig[i] - y_pred_orig[i-1]) < 1e-6:
            count += 1
        else:
            if count > 1:
                consecutive_counts.append(count)
            count = 1
    if count > 1:
        consecutive_counts.append(count)
    
    if consecutive_counts:
        sns.histplot(consecutive_counts, bins=20, kde=True, ax=axes2[1])
        axes2[1].set_xlabel('Consecutive Same Value Count')
        axes2[1].set_ylabel('Frequency')
        axes2[1].set_title('Distribution of Plateau Lengths')
    else:
        axes2[1].text(0.5, 0.5, 'No plateaus detected', ha='center', va='center', fontsize=14)
        axes2[1].set_xlabel('Consecutive Same Value Count')
        axes2[1].set_ylabel('Frequency')
        axes2[1].set_title('Distribution of Plateau Lengths')
    
    # Add title to the figure
    fig.suptitle(f'Smoothing Comparison - {model_name}', fontsize=16, y=0.98)
    fig2.suptitle(f'Smoothing Analysis - {model_name}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    # Print summary comparison
    print("\nSmoothing Approaches Comparison:")
    print(f"  No Smoothing:     RMSE = {rmse_orig:.6f}, R² = {r2_orig:.6f}")
    print(f"  Medium Smoothing: RMSE = {rmse_medium:.6f}, R² = {r2_medium:.6f} (ΔRMSE = {rmse_medium-rmse_orig:.6f})")
    print(f"  High Smoothing:   RMSE = {rmse_high:.6f}, R² = {r2_high:.6f} (ΔRMSE = {rmse_high-rmse_orig:.6f})")
    
    # Calculate plateau statistics
    if plateaus:
        plateau_lengths = [end-start+1 for start, end in plateaus]
        print(f"\nPlateau Statistics:")
        print(f"  Number of plateaus detected: {len(plateaus)}")
        print(f"  Average plateau length: {np.mean(plateau_lengths):.2f} steps")
        print(f"  Longest plateau: {max(plateau_lengths)} steps")
        print(f"  Total steps in plateaus: {sum(plateau_lengths)} ({100*sum(plateau_lengths)/len(y_pred_orig):.2f}% of series)")
    else:
        print("\nNo significant plateaus detected in the prediction.")
    
    return rmse_orig, rmse_medium, rmse_high, r2_orig, r2_medium, r2_high


def main():
    """Main execution function"""
    print("SPS Sintering Regression Analysis")
    print(f"Running approach {APPROACH}")

    # Load data
    print("\nLoading data...")
    train_data, validation_data = load_data(file_paths, VALIDATION_FILE_INDEX)
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    
    # Analyze precision issues in the target column
    all_data = pd.concat([train_data, validation_data])
    precision_results = analyze_target_precision(all_data, TARGET_COLUMN)

    # Preprocess data based on the selected approach
    print("\nPreprocessing data...")
    if APPROACH == 1:
        # Standard approach: predict target based on current features only
        X_train, y_train, feature_names = preprocess_data(
            train_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)
        X_val, y_val, _ = preprocess_data(
            validation_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)

        # Split training data into train and test sets
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)

        # Build and evaluate models
        print("\nTraining and evaluating models...")
        results, best_model, all_models, scaler = build_and_evaluate_models(
            X_train_split, y_train_split, X_test, y_test)

        # Evaluate best model on validation data
        X_val_scaled = scaler.transform(X_val)
        val_metrics, y_val_pred = evaluate_model(best_model, X_val_scaled, y_val,
                                                 results[0]['model'] + " (Validation)")

        print("\nBest model performance on validation data:")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  R²: {val_metrics['r2']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")

        # Plot results for best model
        best_model_name = results[0]['model']

        # Plot feature importance if available
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            plot_feature_importance(best_model, feature_names, best_model_name)

        # Plot actual vs predicted
        plot_actual_vs_predicted(y_val, y_val_pred, best_model_name, APPROACH)

        # Plot time series prediction
        plot_time_series_prediction(y_val, y_val_pred, best_model_name, APPROACH)

        # Plot residuals
        plot_residuals(y_val, y_val_pred, best_model_name, APPROACH)

    elif APPROACH == 2:
        # Window approach: use previous step data and target to predict next step
        # First preprocess the data without windowing
        X_train, y_train, feature_names = preprocess_data(
            train_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)
        X_val, y_val, _ = preprocess_data(
            validation_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)

        # Create windowed data (include one previous step)
        window_size = 1
        X_train_window, y_train_window = prepare_window_data(X_train, y_train, window_size)
        X_val_window, y_val_window = prepare_window_data(X_val, y_val, window_size)

        # Create feature names for windowed data
        window_feature_names = []
        for w in range(window_size + 1):
            prefix = "" if w == 0 else f"prev{w}_"
            window_feature_names.extend([f"{prefix}{name}" for name in feature_names])
        for w in range(1, window_size + 1):
            window_feature_names.append(f"prev{w}_{TARGET_COLUMN}")

        # Split training data into train and test sets
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train_window, y_train_window, test_size=0.2, random_state=42)

        # Build and evaluate models
        print("\nTraining and evaluating models (window approach)...")
        results, best_model, all_models, scaler = build_and_evaluate_models(
            X_train_split, y_train_split, X_test, y_test)

        # Evaluate best model on validation data
        X_val_scaled = scaler.transform(X_val_window)
        val_metrics, y_val_pred = evaluate_model(best_model, X_val_scaled, y_val_window,
                                                 results[0]['model'] + " (Validation)")

        print("\nBest model performance on validation data (window approach):")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  R²: {val_metrics['r2']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")

        # Plot results for best model
        best_model_name = results[0]['model']

        # Plot feature importance if available
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            plot_feature_importance(best_model, window_feature_names, best_model_name)

        # Plot actual vs predicted
        plot_actual_vs_predicted(y_val_window, y_val_pred, best_model_name, APPROACH)

        # Plot time series prediction
        plot_time_series_prediction(y_val_window, y_val_pred, best_model_name, APPROACH)

        # Plot residuals
        plot_residuals(y_val_window, y_val_pred, best_model_name, APPROACH)

    elif APPROACH == 3:
        # Virtual experiment: similar to approach 2, but using predicted targets
        # for subsequent predictions instead of actual targets

        # First preprocess the data without windowing
        X_train, y_train, feature_names = preprocess_data(
            train_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)
        X_val, y_val, _ = preprocess_data(
            validation_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)

        # Create windowed data (include one previous step)
        window_size = 1
        X_train_window, y_train_window = prepare_window_data(X_train, y_train, window_size)
        X_val_window, y_val_window = prepare_window_data(X_val, y_val, window_size)

        # Create feature names for windowed data
        window_feature_names = []
        for w in range(window_size + 1):
            prefix = "" if w == 0 else f"prev{w}_"
            window_feature_names.extend([f"{prefix}{name}" for name in feature_names])
        for w in range(1, window_size + 1):
            window_feature_names.append(f"prev{w}_{TARGET_COLUMN}")

        # Split training data into train and test sets
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train_window, y_train_window, test_size=0.2, random_state=42)

        # Build and evaluate models
        print("\nTraining and evaluating models (window approach)...")
        results, best_model, all_models, scaler = build_and_evaluate_models(
            X_train_split, y_train_split, X_test, y_test)

        # Run virtual experiment with the best model
        print("\nRunning virtual experiment...")
        # Scale the validation data
        X_val_window_scaled = scaler.transform(X_val_window)

        # Regular window approach prediction for comparison
        val_metrics, y_val_window_pred = evaluate_model(
            best_model, X_val_window_scaled, y_val_window,
            results[0]['model'] + " (Window Approach)")

        # Virtual experiment approach
        y_virtual_pred, y_virtual_true = virtual_experiment(
            best_model, X_val_window_scaled, y_val, window_feature_names, window_size)

        # Calculate metrics for virtual experiment
        virtual_mse = mean_squared_error(y_virtual_true, y_virtual_pred)
        virtual_rmse = np.sqrt(virtual_mse)
        virtual_mae = mean_absolute_error(y_virtual_true, y_virtual_pred)
        virtual_r2 = r2_score(y_virtual_true, y_virtual_pred)

        print("\nVirtual experiment results:")
        print(f"  RMSE: {virtual_rmse:.4f}")
        print(f"  R²: {virtual_r2:.4f}")
        print(f"  MAE: {virtual_mae:.4f}")

        # Compare with window approach
        print("\nComparison with window approach:")
        print(f"  Window approach RMSE: {val_metrics['rmse']:.4f}, Virtual experiment RMSE: {virtual_rmse:.4f}")
        print(f"  Window approach R²: {val_metrics['r2']:.4f}, Virtual experiment R²: {virtual_r2:.4f}")

        # Plot results for virtual experiment
        best_model_name = results[0]['model']

        # Plot actual vs predicted
        plot_actual_vs_predicted(y_virtual_true, y_virtual_pred, best_model_name, APPROACH)

        # Plot time series prediction
        plot_time_series_prediction(y_virtual_true, y_virtual_pred, best_model_name, APPROACH)

        # Plot residuals
        plot_residuals(y_virtual_true, y_virtual_pred, best_model_name, APPROACH)

    else:
        raise ValueError(f"Unknown approach: {APPROACH}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()