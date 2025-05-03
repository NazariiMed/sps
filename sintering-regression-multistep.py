import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
import xgboost as xgb
import warnings
import time

# Import tuning functionality from shared module
from sintering_tuning import tune_hyperparameters, get_param_grids, ensure_finite

# Try to import tqdm for progress bars, but continue if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple alternative to tqdm
    def tqdm(iterable, desc=None):
        print(f"{desc if desc else 'Progress'}...")
        return iterable


# ensure_finite function is now imported from sintering_tuning.py

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

# Configuration
VALIDATION_FILE_INDEX = 3  # Use the 4th file for validation (0-indexed)
TARGET_COLUMN = 'Rel. Piston Trav'
EXCLUDED_COLUMNS = ['Abs. Piston Trav', 'Nr.', 'Datum', 'Zeit']

# Feature selection (manual control)
SELECTED_FEATURES = [
    'MTC1', 'MTC2', 'MTC3', 'Pyrometer', 'SV Temperature',
    'SV Power', 'SV Force', 'AV Force', 'AV Rel. Pressure',
    'I RMS', 'U RMS', 'Heating power'
]

# Model selection (we'll focus on a subset for the multi-step training)
MODELS_TO_EVALUATE = {
    'Linear Regression': True,
    'Ridge': True,
    'Random Forest': True, 
    'Gradient Boosting': True,
    'XGBoost': True
}

# Hyperparameter tuning settings
TUNING_METHOD = 'random'  # 'grid', 'random', 'bayesian'
CV_FOLDS = 3  # Reduced from 5 for faster training
N_ITER = 10  # Reduced from 20 for faster training
USE_OPTIMIZED_MODELS = True  # Whether to use hyperparameter-optimized models

# Multi-step training parameters
WINDOW_SIZE = 1
MAX_EPOCHS = 10
CURRICULUM_STEPS = [1, 2, 5, 10, 20, 50, 100]  # Gradually increase prediction length
TEACHER_FORCING_RATIO_START = 1.0  # Start with 100% ground truth
TEACHER_FORCING_RATIO_END = 0.0   # End with 0% ground truth (all predictions)
BATCH_SIZE = 128


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
        target_col: Name of the target column
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
    y = y.astype(np.float64)

    print(f"Preprocessed data: X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_names


def prepare_window_data(X, y, window_size=1):
    """
    Prepare data for window-based approach.

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


def create_batches(X, y, batch_size):
    """
    Create batches from data.
    
    Args:
        X: Feature matrix
        y: Target vector
        batch_size: Size of each batch
        
    Returns:
        batches: List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches


def create_sequences(X_window, y_window, seq_len, n_features_per_window, window_size):
    """
    Create sequences for sequential prediction.
    
    Args:
        X_window: Windowed feature matrix
        y_window: Corresponding target values
        seq_len: Length of each sequence
        n_features_per_window: Number of features per window
        window_size: Window size
        
    Returns:
        sequences: List of (features, targets) tuples
    """
    n_samples = X_window.shape[0]
    
    sequences = []
    for i in range(n_samples - seq_len + 1):
        # Extract sequence of inputs and outputs
        X_seq = X_window[i:i+seq_len]
        y_seq = y_window[i:i+seq_len]
        
        sequences.append((X_seq, y_seq))
    
    return sequences


def virtual_experiment_predict(model, X_input, y_prev_actual, n_features_per_window, window_size, 
                              use_predictions=False, prev_prediction=None):
    """
    Make predictions using either actual values or previous predictions.
    
    Args:
        model: Trained model
        X_input: Current input features 
        y_prev_actual: Previous actual target values
        n_features_per_window: Number of features per window
        window_size: Window size
        use_predictions: Whether to use model's predictions instead of actual values
        prev_prediction: Previous prediction (if use_predictions is True)
        
    Returns:
        prediction: Model's prediction
    """
    # Make a copy to avoid modifying the original
    X_modified = X_input.copy()
    
    # If using predictions, replace the previous target value in the input
    if use_predictions and prev_prediction is not None:
        # Find the index of the previous target value
        prev_target_idx = n_features_per_window * (window_size + 1) + (window_size - 1)
        X_modified[prev_target_idx] = prev_prediction
    
    # Ensure finite values
    X_modified = ensure_finite(X_modified)
    
    # Make prediction - ensure 2D array (samples, features)
    try:
        prediction = model.predict(X_modified.reshape(1, -1))[0]
        
        # Ensure prediction is finite
        if not np.isfinite(prediction):
            print("Warning: Non-finite prediction detected, using default value.")
            if prev_prediction is not None:
                prediction = prev_prediction  # Use previous prediction as fallback
            else:
                prediction = 0.0  # Default fallback
    except Exception as e:
        print(f"Error making prediction: {e}")
        # Fallback to a reasonable value
        if prev_prediction is not None:
            prediction = prev_prediction
        else:
            prediction = 0.0
    
    return prediction


def create_base_models(X_train, y_train, use_optimized=True):
    """
    Create base models for training with optional hyperparameter optimization.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        use_optimized: Whether to optimize hyperparameters
    
    Returns:
        models: Dictionary of model instances
    """
    models = {}
    
    # Get parameter grids for tuning
    param_grids, param_ranges = get_param_grids()
    
    # Create and potentially tune models
    if MODELS_TO_EVALUATE.get('Linear Regression', False):
        if use_optimized:
            print("Tuning Linear Regression...")
            model_class = LinearRegression
            # Select parameter grid based on tuning method
            param_config = param_ranges['Linear Regression'] if TUNING_METHOD == 'bayesian' else param_grids['Linear Regression']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='Linear Regression')
            models['Linear Regression'] = model
        else:
            models['Linear Regression'] = LinearRegression()
    
    if MODELS_TO_EVALUATE.get('Ridge', False):
        if use_optimized:
            print("Tuning Ridge...")
            model_class = Ridge
            param_config = param_ranges['Ridge'] if TUNING_METHOD == 'bayesian' else param_grids['Ridge']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='Ridge')
            models['Ridge'] = model
        else:
            models['Ridge'] = Ridge(alpha=1.0)
    
    if MODELS_TO_EVALUATE.get('Lasso', False):
        if use_optimized:
            print("Tuning Lasso...")
            model_class = Lasso
            param_config = param_ranges['Lasso'] if TUNING_METHOD == 'bayesian' else param_grids['Lasso']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='Lasso')
            models['Lasso'] = model
        else:
            models['Lasso'] = Lasso(alpha=0.01)
    
    if MODELS_TO_EVALUATE.get('ElasticNet', False):
        if use_optimized:
            print("Tuning ElasticNet...")
            model_class = ElasticNet
            param_config = param_ranges['ElasticNet'] if TUNING_METHOD == 'bayesian' else param_grids['ElasticNet']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='ElasticNet')
            models['ElasticNet'] = model
        else:
            models['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.5)
    
    if MODELS_TO_EVALUATE.get('Random Forest', False):
        if use_optimized:
            print("Tuning Random Forest...")
            model_class = RandomForestRegressor
            param_config = param_ranges['Random Forest'] if TUNING_METHOD == 'bayesian' else param_grids['Random Forest']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='Random Forest')
            models['Random Forest'] = model
        else:
            models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    
    if MODELS_TO_EVALUATE.get('Gradient Boosting', False):
        if use_optimized:
            print("Tuning Gradient Boosting...")
            model_class = GradientBoostingRegressor
            param_config = param_ranges['Gradient Boosting'] if TUNING_METHOD == 'bayesian' else param_grids['Gradient Boosting']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='Gradient Boosting')
            models['Gradient Boosting'] = model
        else:
            models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    if MODELS_TO_EVALUATE.get('XGBoost', False):
        if use_optimized:
            print("Tuning XGBoost...")
            model_class = xgb.XGBRegressor
            param_config = param_ranges['XGBoost'] if TUNING_METHOD == 'bayesian' else param_grids['XGBoost']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='XGBoost')
            models['XGBoost'] = model
        else:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    if MODELS_TO_EVALUATE.get('SVR', False):
        if use_optimized:
            print("Tuning SVR...")
            model_class = SVR
            param_config = param_ranges['SVR'] if TUNING_METHOD == 'bayesian' else param_grids['SVR']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='SVR')
            models['SVR'] = model
        else:
            models['SVR'] = SVR(kernel='rbf', C=10)
    
    if MODELS_TO_EVALUATE.get('KNN', False):
        if use_optimized:
            print("Tuning KNN...")
            model_class = KNeighborsRegressor
            param_config = param_ranges['KNN'] if TUNING_METHOD == 'bayesian' else param_grids['KNN']
            model, _ = tune_hyperparameters(model_class, param_config, X_train, y_train,
                                           method=TUNING_METHOD, cv=CV_FOLDS, n_iter=N_ITER,
                                           model_name='KNN')
            models['KNN'] = model
        else:
            models['KNN'] = KNeighborsRegressor(n_neighbors=5)
    
    return models


def multi_step_train(model, X_train, y_train, X_val, y_val, n_features_per_window, window_size, 
                    max_epochs=10, curriculum_steps=None, tf_ratio_start=1.0, tf_ratio_end=0.0, 
                    batch_size=128, verbose=True):
    """
    Train a model using multi-step approach with scheduled sampling.
    
    Args:
        model: Base model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_features_per_window: Number of features per window
        window_size: Window size
        max_epochs: Maximum number of epochs to train
        curriculum_steps: List of sequence lengths for curriculum learning
        tf_ratio_start: Initial teacher forcing ratio (1.0 = always use ground truth)
        tf_ratio_end: Final teacher forcing ratio (0.0 = always use predictions)
        batch_size: Batch size for training
        verbose: Whether to print progress
        
    Returns:
        trained_model: Trained model
        history: Training history
    """
    # Clone the model to start fresh
    trained_model = clone(model)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'tf_ratio': []
    }
    
    # Default curriculum if not provided
    if curriculum_steps is None:
        curriculum_steps = [1, 2, 5, 10, 20]
    
    # Train with curriculum learning (increasing sequence length)
    for step_idx, seq_len in enumerate(curriculum_steps):
        print(f"\nTraining with sequence length: {seq_len}")
        
        # Calculate effective number of epochs for this step
        step_epochs = max(1, int(max_epochs / len(curriculum_steps)))
        
        # Create training sequences
        train_sequences = create_sequences(X_train, y_train, seq_len, n_features_per_window, window_size)
        
        # Train for multiple epochs
        for epoch in range(step_epochs):
            # Calculate current teacher forcing ratio
            progress = (step_idx * step_epochs + epoch) / (len(curriculum_steps) * step_epochs)
            tf_ratio = tf_ratio_start - progress * (tf_ratio_start - tf_ratio_end)
            history['tf_ratio'].append(tf_ratio)
            
            print(f"Epoch {epoch+1}/{step_epochs}, Teacher Forcing Ratio: {tf_ratio:.3f}")
            
            # Shuffle sequences for this epoch
            np.random.shuffle(train_sequences)
            
            # Process sequences in batches
            train_losses = []
            train_r2s = []
            
            n_batches = (len(train_sequences) + batch_size - 1) // batch_size
            
            # Use tqdm if available, otherwise use simple progress updates
            batch_range = tqdm(range(n_batches), desc="Training") if TQDM_AVAILABLE else range(n_batches)
            if not TQDM_AVAILABLE and n_batches > 10:
                print(f"Processing {n_batches} batches...")
                
            for batch_idx in batch_range:
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(train_sequences))
                batch_sequences = train_sequences[batch_start:batch_end]
                
                batch_losses = []
                batch_r2s = []
                
                for X_seq, y_seq in batch_sequences:
                    seq_predictions = []
                    seq_actuals = []
                    prev_prediction = None
                    
                    # Process each step in the sequence
                    for step in range(len(X_seq)):
                        # Decide whether to use ground truth or prediction
                        use_prediction = (step > 0) and (np.random.random() > tf_ratio)
                        
                        # Make prediction
                        try:
                            prediction = virtual_experiment_predict(
                                trained_model, X_seq[step], y_seq[:step] if step > 0 else [],
                                n_features_per_window, window_size,
                                use_prediction, prev_prediction
                            )
                        except Exception as e:
                            print(f"Error making prediction: {e}")
                            # Fallback: use simple prediction without modifications
                            prediction = trained_model.predict(X_seq[step].reshape(1, -1))[0]
                        
                        # Store prediction and actual values
                        seq_predictions.append(prediction)
                        seq_actuals.append(y_seq[step])
                        
                        # Update for next step
                        prev_prediction = prediction
                    
                    # Calculate loss for this sequence
                    mse = mean_squared_error(seq_actuals, seq_predictions)
                    batch_losses.append(mse)
                    
                    # Calculate R² for this sequence
                    r2 = r2_score(seq_actuals, seq_predictions)
                    batch_r2s.append(r2)
                
                # Update model based on batch losses
                if hasattr(trained_model, 'partial_fit'):
                    # For models that support incremental learning
                    for X_seq, y_seq in batch_sequences:
                        trained_model.partial_fit(X_seq, y_seq)
                else:
                    # For models that require full batch training
                    # Gather all training data from this batch with scheduled sampling
                    X_batch = []
                    y_batch = []
                    
                    for X_seq, y_seq in batch_sequences:
                        # Process each sequence and collect inputs/outputs with teacher forcing
                        seq_X = []
                        seq_y = []
                        seq_preds = []
                        
                        for step in range(len(X_seq)):
                            X_modified = X_seq[step].copy()
                            
                            # Apply teacher forcing for previous step's target if needed
                            if step > 0 and np.random.random() > tf_ratio:
                                # Use prediction for previous step
                                prev_target_idx = n_features_per_window * (window_size + 1) + (window_size - 1)
                                # Handle edge case where seq_preds might be empty
                                if seq_preds:
                                    X_modified[prev_target_idx] = seq_preds[-1]
                            
                            # Clean input to ensure finite values
                            X_modified = ensure_finite(X_modified)
                            
                            # Make prediction for this step (for next step's input if needed)
                            try:
                                pred = trained_model.predict(X_modified.reshape(1, -1))[0]
                                
                                # Ensure prediction is finite
                                if not np.isfinite(pred):
                                    print("Warning: Non-finite batch prediction detected, using actual value.")
                                    pred = y_seq[step]  # Use actual value as fallback
                                    
                                seq_preds.append(pred)
                            except Exception as e:
                                print(f"Error in batch prediction: {e}")
                                # If prediction fails, use actual value
                                seq_preds.append(y_seq[step])
                            
                            seq_X.append(X_modified)
                            seq_y.append(y_seq[step])
                        
                        # Add this sequence's data to the batch
                        X_batch.extend(seq_X)
                        y_batch.extend(seq_y)
                    
                    # Convert to numpy arrays
                    X_batch = np.array(X_batch)
                    y_batch = np.array(y_batch)
                    
                    # Update the model (only if we have data)
                    if len(X_batch) > 0:
                        try:
                            trained_model.fit(X_batch, y_batch)
                        except Exception as e:
                            print(f"Error fitting model: {e}")
                            # If batch fitting fails, try individual fitting
                            for i in range(len(X_batch)):
                                try:
                                    trained_model.partial_fit(X_batch[i:i+1], y_batch[i:i+1])
                                except:
                                    pass  # Skip if partial_fit isn't available
                
                # Track batch metrics
                avg_batch_loss = np.mean(batch_losses)
                avg_batch_r2 = np.mean(batch_r2s)
                train_losses.append(avg_batch_loss)
                train_r2s.append(avg_batch_r2)
            
            # Calculate overall metrics for this epoch
            epoch_train_loss = np.mean(train_losses)
            epoch_train_r2 = np.mean(train_r2s)
            
            # Evaluate on validation data
            val_predictions = []
            val_actuals = []
            
            # Create validation sequences
            val_sequences = create_sequences(X_val, y_val, seq_len, n_features_per_window, window_size)
            
            # Process validation sequences
            for X_seq, y_seq in val_sequences[:100]:  # Limit to 100 sequences for speed
                seq_predictions = []
                prev_prediction = None
                
                for step in range(len(X_seq)):
                    # Always use previous predictions for validation
                    use_prediction = step > 0
                    
                    try:
                        prediction = virtual_experiment_predict(
                            trained_model, X_seq[step], y_seq[:step] if step > 0 else [],
                            n_features_per_window, window_size,
                            use_prediction, prev_prediction
                        )
                    except Exception as e:
                        print(f"Error in validation prediction: {e}")
                        # Fallback: use simple prediction without modifications
                        prediction = trained_model.predict(X_seq[step].reshape(1, -1))[0]
                    
                    seq_predictions.append(prediction)
                    prev_prediction = prediction
                
                val_predictions.extend(seq_predictions)
                val_actuals.extend(y_seq)
            
            # Calculate validation metrics
            epoch_val_loss = mean_squared_error(val_actuals, val_predictions)
            epoch_val_r2 = r2_score(val_actuals, val_predictions)
            
            # Record history
            history['train_loss'].append(epoch_train_loss)
            history['train_r2'].append(epoch_train_r2)
            history['val_loss'].append(epoch_val_loss)
            history['val_r2'].append(epoch_val_r2)
            
            print(f"  Train Loss: {epoch_train_loss:.6f}, R²: {epoch_train_r2:.4f}")
            print(f"  Val Loss: {epoch_val_loss:.6f}, R²: {epoch_val_r2:.4f}")
    
    return trained_model, history


def virtual_experiment(model, X_val, y_val, n_features_per_window, window_size=1):
    """
    Run a virtual experiment with the trained model.

    Args:
        model: Trained regression model
        X_val: Validation feature matrix
        y_val: Validation target vector
        n_features_per_window: Number of features per window
        window_size: Window size used in training

    Returns:
        y_pred: Predicted target values
        y_true: Actual target values
    """
    n_samples = X_val.shape[0]

    # We'll store actual and predicted values
    y_true = []
    y_pred = []

    # Initialize with the first few actual values
    prev_y_values = y_val[:window_size].tolist()

    # Process each time step
    for i in range(window_size, n_samples):
        # Extract current input
        X_input = X_val[i].copy()  # Make a copy to avoid modifying the original

        # Use previous predictions instead of actual values
        for w in range(1, window_size + 1):
            prev_y_idx = n_features_per_window * (window_size + 1) + (w - 1)
            X_input[prev_y_idx] = prev_y_values[-w]

        # Clean input data to ensure finite values
        X_input = ensure_finite(X_input)
            
        # Make prediction - reshape to 2D array (samples, features)
        try:
            prediction = model.predict(X_input.reshape(1, -1))[0]
            
            # Ensure prediction is finite
            if not np.isfinite(prediction):
                print("Warning: Non-finite prediction detected, using default value.")
                prediction = prev_y_values[-1]  # Use previous value as fallback
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            # Fallback: use previous value as prediction
            prediction = prev_y_values[-1]

        # Store actual and predicted values
        y_true.append(y_val[i])
        y_pred.append(prediction)

        # Update previous y values for next iteration
        prev_y_values.append(prediction)

    return np.array(y_pred), np.array(y_true)


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
        y_pred: Predicted values
    """
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


def plot_learning_curves(history, model_name):
    """
    Plot learning curves from training history.

    Args:
        history: Training history dictionary
        model_name: Name of the model
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Loss curves
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: R² curves
    axes[1].plot(history['train_r2'], label='Training R²')
    axes[1].plot(history['val_r2'], label='Validation R²')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Teacher forcing ratio
    axes[2].plot(history['tf_ratio'], label='Teacher Forcing Ratio')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Ratio')
    axes[2].set_title('Teacher Forcing Ratio')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Learning Curves - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    plt.close()


def plot_time_series_prediction(y_true, y_pred, model_name, title=None):
    """
    Plot time series of actual and predicted values.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name: Name of the model
        title: Custom title (optional)
    """
    plt.figure(figsize=(15, 8))

    # Create time series plot
    plt.plot(range(len(y_true)), y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', alpha=0.7)

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Rel. Piston Trav')
    if title:
        plt.title(title)
    else:
        plt.title(f'Time Series Prediction - {model_name}')
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


def plot_actual_vs_predicted(y_true, y_pred, model_name, title=None):
    """
    Plot actual vs predicted values.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name: Name of the model
        title: Custom title (optional)
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
    if title:
        plt.title(title)
    else:
        plt.title(f'Actual vs Predicted - {model_name}')

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


def plot_model_comparison(standard_metrics, multistep_metrics, approach="Virtual Experiment"):
    """
    Plot comparison between standard and multi-step trained models.

    Args:
        standard_metrics: Dictionary of metrics for standard models
        multistep_metrics: Dictionary of metrics for multi-step models
        approach: The approach name for the title
    """
    # Get all model names
    model_names = sorted(set(list(standard_metrics.keys()) + list(multistep_metrics.keys())))
    
    # Prepare data for plotting
    rmse_standard = [standard_metrics.get(name, {}).get('rmse', 0) for name in model_names]
    rmse_multistep = [multistep_metrics.get(name, {}).get('rmse', 0) for name in model_names]
    
    r2_standard = [standard_metrics.get(name, {}).get('r2', 0) for name in model_names]
    r2_multistep = [multistep_metrics.get(name, {}).get('r2', 0) for name in model_names]
    
    # Compute improvement percentages
    rmse_improvement = []
    r2_improvement = []
    
    for i, name in enumerate(model_names):
        if name in standard_metrics and name in multistep_metrics:
            std_rmse = standard_metrics[name]['rmse']
            ms_rmse = multistep_metrics[name]['rmse']
            rmse_imp = ((std_rmse - ms_rmse) / std_rmse) * 100
            rmse_improvement.append(rmse_imp)
            
            std_r2 = standard_metrics[name]['r2']
            ms_r2 = multistep_metrics[name]['r2']
            r2_imp = ((ms_r2 - std_r2) / abs(std_r2)) * 100 if std_r2 != 0 else 0
            r2_improvement.append(r2_imp)
        else:
            rmse_improvement.append(0)
            r2_improvement.append(0)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 14))
    
    # Plot 1: RMSE comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, rmse_standard, width, label='Standard Training', color='red', alpha=0.7)
    axes[0].bar(x + width/2, rmse_multistep, width, label='Multi-step Training', color='green', alpha=0.7)
    
    # Add labels and annotations
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('RMSE (lower is better)')
    axes[0].set_title('RMSE Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    
    # Add value labels on the bars
    for i, (v1, v2, imp) in enumerate(zip(rmse_standard, rmse_multistep, rmse_improvement)):
        if v1 > 0 and v2 > 0:  # Only label bars with valid data
            axes[0].text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0].text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add improvement percentage
            if imp != 0:
                color = 'green' if imp > 0 else 'red'
                axes[0].text(i, min(v1, v2) / 2, f'{imp:.1f}%', ha='center', va='center', 
                          fontsize=9, fontweight='bold', color=color,
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 2: R² comparison
    axes[1].bar(x - width/2, r2_standard, width, label='Standard Training', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, r2_multistep, width, label='Multi-step Training', color='purple', alpha=0.7)
    
    # Add labels and annotations
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('R² (higher is better)')
    axes[1].set_title('R² Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].legend()
    
    # Add value labels on the bars
    for i, (v1, v2, imp) in enumerate(zip(r2_standard, r2_multistep, r2_improvement)):
        if v1 != 0 or v2 != 0:  # Only label bars with valid data
            axes[1].text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=8)
            axes[1].text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add improvement percentage
            if imp != 0:
                color = 'green' if imp > 0 else 'red'
                y_pos = max(0.1, (v1 + v2) / 2)
                axes[1].text(i, y_pos, f'{imp:.1f}%', ha='center', va='center', 
                          fontsize=9, fontweight='bold', color=color,
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add horizontal line at 0 for R² reference
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Standard vs Multi-step Training Comparison ({approach})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    plt.close()


def main():
    """Main execution function"""
    start_time = time.time()
    print("SPS Sintering Multi-step Regression Analysis")
    
    # Create output directory for results
    import os
    results_dir = "multistep_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_data, validation_data = load_data(file_paths, VALIDATION_FILE_INDEX)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train, feature_names = preprocess_data(
        train_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)
    X_val, y_val, _ = preprocess_data(
        validation_data, TARGET_COLUMN, EXCLUDED_COLUMNS, SELECTED_FEATURES)
    
    # Create windowed data
    print("\nCreating windowed data...")
    X_train_window, y_train_window = prepare_window_data(X_train, y_train, WINDOW_SIZE)
    X_val_window, y_val_window = prepare_window_data(X_val, y_val, WINDOW_SIZE)
    
    # Split data for initial training
    print("\nSplitting data...")
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train_window, y_train_window, test_size=0.2, random_state=42)
    
    # Scale the data
    print("\nScaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val_window)
    
    # Train models with standard approach first
    print("\nTraining models with standard approach...")
    
    model_status = "optimized" if USE_OPTIMIZED_MODELS else "default"
    print(f"Using {model_status} hyperparameters")
    
    base_models = create_base_models(X_train_scaled, y_train_split, USE_OPTIMIZED_MODELS)
    standard_models = {}
    standard_metrics = {}
    
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        if not USE_OPTIMIZED_MODELS:  # If we're using default models, we need to fit them here
            model.fit(X_train_scaled, y_train_split)
        
        # Evaluate on test set
        test_metrics, _ = evaluate_model(model, X_test_scaled, y_test, name)
        print(f"  Test - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        # Run virtual experiment
        print(f"Running virtual experiment for {name}...")
        y_virtual_pred, y_virtual_true = virtual_experiment(
            model, X_val_scaled, y_val_window, X_train.shape[1], WINDOW_SIZE)
        
        # Calculate metrics
        virtual_mse = mean_squared_error(y_virtual_true, y_virtual_pred)
        virtual_rmse = np.sqrt(virtual_mse)
        virtual_r2 = r2_score(y_virtual_true, y_virtual_pred)
        
        print(f"  Virtual - RMSE: {virtual_rmse:.4f}, R²: {virtual_r2:.4f}")
        
        # Store model and metrics
        standard_models[name] = model
        standard_metrics[name] = {
            'rmse': virtual_rmse,
            'r2': virtual_r2
        }
        
        # Plot results
        plot_time_series_prediction(
            y_virtual_true, y_virtual_pred, name, 
            title=f"Virtual Experiment - {name} (Standard {model_status} Training)"
        )
    
    # Train models with multi-step approach
    print("\nTraining models with multi-step approach...")
    n_features_per_window = X_train.shape[1]
    multistep_models = {}
    multistep_metrics = {}
    training_histories = {}
    
    for name, base_model in base_models.items():
        print(f"\nMulti-step training for {name}...")
        
        # Use a fresh model instance
        model_to_train = clone(base_model)
        
        # First fit with standard approach to have a starting point
        if not USE_OPTIMIZED_MODELS:  # Only need to refit if we're using default models
            model_to_train.fit(X_train_scaled, y_train_split)
        
        # Then apply multi-step training
        trained_model, history = multi_step_train(
            model_to_train, X_train_scaled, y_train_split, X_val_scaled, y_val_window,
            n_features_per_window, WINDOW_SIZE,
            max_epochs=MAX_EPOCHS, 
            curriculum_steps=CURRICULUM_STEPS,
            tf_ratio_start=TEACHER_FORCING_RATIO_START,
            tf_ratio_end=TEACHER_FORCING_RATIO_END,
            batch_size=BATCH_SIZE,
            verbose=True
        )
        
        # Run virtual experiment with multi-step trained model
        print(f"Running virtual experiment for multi-step trained {name}...")
        y_ms_virtual_pred, y_ms_virtual_true = virtual_experiment(
            trained_model, X_val_scaled, y_val_window, n_features_per_window, WINDOW_SIZE)
        
        # Calculate metrics
        ms_virtual_mse = mean_squared_error(y_ms_virtual_true, y_ms_virtual_pred)
        ms_virtual_rmse = np.sqrt(ms_virtual_mse)
        ms_virtual_r2 = r2_score(y_ms_virtual_true, y_ms_virtual_pred)
        
        print(f"  Multi-step Virtual - RMSE: {ms_virtual_rmse:.4f}, R²: {ms_virtual_r2:.4f}")
        
        # Store model, metrics, and history
        multistep_models[name] = trained_model
        multistep_metrics[name] = {
            'rmse': ms_virtual_rmse,
            'r2': ms_virtual_r2
        }
        training_histories[name] = history
        
        # Plot results
        plot_learning_curves(history, name)
        
        plot_time_series_prediction(
            y_ms_virtual_true, y_ms_virtual_pred, name, 
            title=f"Virtual Experiment - {name} (Multi-step {model_status} Training)"
        )
        
        # Compare standard vs multi-step for this model
        if name in standard_metrics:
            # Get predictions from both models
            y_std_pred, _ = virtual_experiment(
                standard_models[name], X_val_scaled, y_val_window, 
                n_features_per_window, WINDOW_SIZE)
            
            # Plot comparison
            plt.figure(figsize=(15, 8))
            plt.plot(y_virtual_true, label='Actual', color='blue', alpha=0.7)
            plt.plot(y_std_pred, label=f'Standard (RMSE={standard_metrics[name]["rmse"]:.4f}, R²={standard_metrics[name]["r2"]:.4f})', 
                    color='red', alpha=0.7)
            plt.plot(y_ms_virtual_pred, label=f'Multi-step (RMSE={ms_virtual_rmse:.4f}, R²={ms_virtual_r2:.4f})', 
                     color='green', alpha=0.7)
            
            plt.xlabel('Time Step')
            plt.ylabel('Rel. Piston Trav')
            plt.title(f'Standard vs Multi-step Training Comparison - {name} ({model_status})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()
    
    # Overall comparison between standard and multi-step approaches
    plot_model_comparison(standard_metrics, multistep_metrics)
    
    # Calculate overall improvement if we have data
    if standard_metrics and multistep_metrics:
        std_rmse = np.mean([m['rmse'] for m in standard_metrics.values()])
        ms_rmse = np.mean([m['rmse'] for m in multistep_metrics.values()])
        rmse_improvement = ((std_rmse - ms_rmse) / std_rmse) * 100 if std_rmse != 0 else 0
        
        std_r2 = np.mean([m['r2'] for m in standard_metrics.values()])
        ms_r2 = np.mean([m['r2'] for m in multistep_metrics.values()])
        r2_improvement = ((ms_r2 - std_r2) / abs(std_r2)) * 100 if std_r2 != 0 else 0
        
        print("\nOverall Improvement:")
        print(f"  Standard Training ({model_status})  - Average RMSE: {std_rmse:.4f}, Average R²: {std_r2:.4f}")
        print(f"  Multi-step Training ({model_status}) - Average RMSE: {ms_rmse:.4f}, Average R²: {ms_r2:.4f}")
        print(f"  RMSE Improvement: {rmse_improvement:.2f}%")
        print(f"  R² Improvement: {r2_improvement:.2f}%")
    else:
        print("\nNot enough data to calculate improvement metrics.")
    
    # Save summary of results to file
    summary_file = os.path.join(results_dir, f"summary_{model_status}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"SPS Sintering Multi-step Regression Analysis Summary ({model_status} models)\n\n")
        f.write(f"Standard Training Results:\n")
        for name, metrics in standard_metrics.items():
            f.write(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}\n")
        
        f.write(f"\nMulti-step Training Results:\n")
        for name, metrics in multistep_metrics.items():
            f.write(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}\n")
            
        if standard_metrics and multistep_metrics:
            f.write(f"\nOverall Improvement:\n")
            f.write(f"  Standard Training  - Average RMSE: {std_rmse:.4f}, Average R²: {std_r2:.4f}\n")
            f.write(f"  Multi-step Training - Average RMSE: {ms_rmse:.4f}, Average R²: {ms_r2:.4f}\n")
            f.write(f"  RMSE Improvement: {rmse_improvement:.2f}%\n")
            f.write(f"  R² Improvement: {r2_improvement:.2f}%\n")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    print(f"\nResults saved to {summary_file}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
