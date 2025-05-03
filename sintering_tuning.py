import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Try to import Bayesian optimization libraries
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Bayesian optimization libraries not available. Will use RandomizedSearchCV instead.")


def ensure_finite(X, default_value=0.0):
    """
    Replace any NaN, inf, or extremely large values with a default value.
    
    Args:
        X: Input array or matrix
        default_value: Value to use for replacement
        
    Returns:
        X_clean: Cleaned array with finite values
    """
    # Make a copy to avoid modifying the original
    X_clean = np.array(X, copy=True)
    
    # Replace inf values
    mask_inf = np.isinf(X_clean)
    if np.any(mask_inf):
        print(f"Warning: Found {np.sum(mask_inf)} infinite values. Replacing with {default_value}.")
        X_clean[mask_inf] = default_value
    
    # Replace NaN values
    mask_nan = np.isnan(X_clean)
    if np.any(mask_nan):
        print(f"Warning: Found {np.sum(mask_nan)} NaN values. Replacing with {default_value}.")
        X_clean[mask_nan] = default_value
    
    # Check for extremely large values
    large_threshold = 1e6  # Adjust as needed
    mask_large = np.abs(X_clean) > large_threshold
    if np.any(mask_large):
        print(f"Warning: Found {np.sum(mask_large)} extremely large values. Replacing with {default_value}.")
        X_clean[mask_large] = default_value
        
    return X_clean


def tune_hyperparameters(model_class, param_grid, X_train, y_train, method='grid', cv=5, n_iter=20, model_name=None):
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
        model_name: Name of the model for special handling

    Returns:
        best_model: Tuned model
        best_params: Best hyperparameter values
    """
    start_time = time.time()
    print(f"  Tuning hyperparameters for {model_name} using {method} search...")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    if method == 'grid':
        search = GridSearchCV(
            model_class(), param_grid, cv=cv, scoring='neg_mean_squared_error',
            verbose=1, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

    elif method == 'random':
        search = RandomizedSearchCV(
            model_class(), param_grid, n_iter=n_iter, cv=cv,
            scoring='neg_mean_squared_error', verbose=1, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

    elif method == 'bayesian':
        if BAYESIAN_AVAILABLE:
            # Convert param_grid to skopt space format
            search_space = {}
            for param, values in param_grid.items():
                # If parameter values are a list
                if isinstance(values, list):
                    # Check types of values to determine space type
                    if all(isinstance(v, bool) for v in values) or all(isinstance(v, str) for v in values):
                        search_space[param] = Categorical(values)
                    elif all(isinstance(v, int) for v in values):
                        search_space[param] = Integer(min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        search_space[param] = Real(min(values), max(values), prior='log-uniform')
                    else:
                        # Mixed types or other - use categorical
                        search_space[param] = Categorical(values)
                # If parameter values are already a dictionary or distribution
                else:
                    search_space[param] = values

            print(f"  Created Bayesian search space: {search_space}")

            # Special handling for models that need parameter mapping
            model_instance = model_class()
            if model_name == 'GPR' and 'kernel' in search_space:
                # Create a modified search with a custom kernel mapping
                def map_kernel(params):
                    # Map numeric values to actual kernels
                    if 'kernel' in params and isinstance(params['kernel'], int):
                        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
                        kernel_map = {
                            1: C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
                            2: C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5),
                            3: C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1)
                        }
                        params['kernel'] = kernel_map.get(params['kernel'], kernel_map[1])
                    return params

                # Use a subset of data for GPR to speed up training
                subset_size = min(1000, len(X_train))
                idx = np.random.choice(len(X_train), subset_size, replace=False)
                X_subset = X_train[idx]
                y_subset = y_train[idx]

                # Manual Bayesian optimization for GPR
                best_score = float('-inf')
                best_params = {}
                best_model = None  # Initialize best_model

                for _ in range(n_iter):
                    # Sample parameters randomly from the space
                    params = {}
                    for param, space in search_space.items():
                        if hasattr(space, 'rvs'):  # It's a distribution
                            params[param] = space.rvs(1)[0]
                        elif isinstance(space, list):  # It's a list of values
                            params[param] = np.random.choice(space)

                    # Map parameters for kernels
                    params = map_kernel(params)

                    # Create and fit model with these parameters
                    try:
                        model = model_class(**params)
                        model.fit(X_subset, y_subset)
                        # Score model
                        score = -mean_squared_error(y_subset, model.predict(X_subset))  # Neg MSE
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_model = model
                    except Exception as e:
                        print(f"  Skipping parameters due to error: {e}")
                        continue
                    
                print(f"  Best params: {best_params}")
                if best_model is None:
                    # Fallback if no model was successfully trained
                    print("  No successful model training, using default parameters")
                    best_model = model_class()
                    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
                    kernel_rbf = C(1.0) * RBF(1.0)
                    best_model.set_params(kernel=kernel_rbf, alpha=1e-6)
                    best_model.fit(X_subset, y_subset)
                return best_model, best_params
            elif model_name == 'MLP' and 'hidden_layer_sizes' in search_space:
                # Create a modified search with a custom hidden_layer_sizes mapping
                def map_hidden_layers(params):
                    # Map numeric values to actual tuples for hidden_layer_sizes
                    if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'], (int, float)):
                        # Map integers to hidden layer configurations
                        layer_map = {
                            1: (50,),
                            2: (100,),
                            3: (50, 50),
                            4: (100, 50)
                        }
                        params['hidden_layer_sizes'] = layer_map.get(int(params['hidden_layer_sizes']), (50,))
                    return params

                # Manual optimization for MLP
                best_score = float('-inf')
                best_params = {}
                best_model = None  # Initialize best_model

                for _ in range(n_iter):
                    # Sample parameters randomly from the space
                    params = {}
                    for param, space in search_space.items():
                        if hasattr(space, 'rvs'):  # It's a distribution
                            params[param] = space.rvs(1)[0]
                        elif isinstance(space, list):  # It's a list of values
                            params[param] = np.random.choice(space)

                    # Map parameters for hidden layer sizes
                    params = map_hidden_layers(params)

                    # Create and fit model with these parameters
                    try:
                        model = model_class(**params)
                        model.fit(X_train, y_train)
                        # Score model
                        score = -mean_squared_error(y_train, model.predict(X_train))  # Neg MSE
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_model = model
                    except Exception as e:
                        print(f"  Skipping parameters due to error: {e}")
                        continue

                print(f"  Best params: {best_params}")
                if best_model is None:
                    # Fallback if no model was successfully trained
                    print("  No successful model training, using default parameters")
                    best_model = model_class(random_state=42, max_iter=1000) 
                    best_model.fit(X_train, y_train)
                return best_model, best_params
            else:
                # For other models, use standard BayesSearchCV
                search = BayesSearchCV(
                    model_instance, search_space, n_iter=n_iter, cv=cv,
                    scoring='neg_mean_squared_error', verbose=1, random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            print("  Bayesian optimization not available, falling back to RandomizedSearchCV")
            search = RandomizedSearchCV(
                model_class(), param_grid, n_iter=n_iter, cv=cv,
                scoring='neg_mean_squared_error', verbose=1, random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

    else:
        raise ValueError(f"Unknown tuning method: {method}")

    elapsed_time = time.time() - start_time
    print(f"  Tuning completed in {elapsed_time:.2f} seconds")
    print(f"  Best params: {best_params}")

    return best_model, best_params


def get_param_grids():
    """
    Get parameter grids for different models.
    
    Returns:
        param_grids: Dictionary of parameter grids for grid/random search
        param_ranges: Dictionary of parameter ranges for Bayesian optimization
    """
    # Parameter grids for grid/random search
    param_grids = {}
    
    # Linear models
    param_grids['Linear Regression'] = {'fit_intercept': [True, False]}
    
    param_grids['Ridge'] = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    param_grids['Lasso'] = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 3000, 5000]
    }
    
    param_grids['ElasticNet'] = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'fit_intercept': [True, False],
        'max_iter': [1000, 3000, 5000]
    }
    
    # Tree-based models
    param_grids['Decision Tree'] = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    param_grids['Random Forest'] = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    param_grids['Gradient Boosting'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10]
    }
    
    param_grids['XGBoost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    param_grids['LightGBM'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'num_leaves': [31, 63, 127],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Other models
    param_grids['SVR'] = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    
    param_grids['KNN'] = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    # Neural Network model
    param_grids['MLP'] = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Gaussian Process Regression
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
    kernel_rbf = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    kernel_matern = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5)
    kernel_rbf_white = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1)
    
    param_grids['GPR'] = {
        'kernel': [kernel_rbf, kernel_matern, kernel_rbf_white],
        'alpha': [1e-10, 1e-8, 1e-6],
        'normalize_y': [True, False],
        'n_restarts_optimizer': [0, 1, 3]
    }
    
    # Parameter ranges for Bayesian optimization
    param_ranges = {}
    
    # Linear models
    param_ranges['Linear Regression'] = {'fit_intercept': [True, False]}
    
    param_ranges['Ridge'] = {
        'alpha': (0.001, 100.0, 'log-uniform'),
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    param_ranges['Lasso'] = {
        'alpha': (0.0001, 10.0, 'log-uniform'),
        'fit_intercept': [True, False],
        'max_iter': (1000, 10000)
    }
    
    param_ranges['ElasticNet'] = {
        'alpha': (0.0001, 1.0, 'log-uniform'),
        'l1_ratio': (0.1, 0.9),
        'fit_intercept': [True, False],
        'max_iter': (1000, 10000)
    }
    
    # Tree-based models
    param_ranges['Decision Tree'] = {
        'max_depth': (3, 30),  # None will be handled specially
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    
    param_ranges['Random Forest'] = {
        'n_estimators': (10, 300),
        'max_depth': (3, 50),  # None will be handled specially
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    
    param_ranges['Gradient Boosting'] = {
        'n_estimators': (10, 300),
        'learning_rate': (0.001, 0.3, 'log-uniform'),
        'max_depth': (2, 15),
        'min_samples_split': (2, 20)
    }
    
    param_ranges['XGBoost'] = {
        'n_estimators': (10, 300),
        'learning_rate': (0.001, 0.3, 'log-uniform'),
        'max_depth': (2, 15),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }
    
    # Other models
    param_ranges['SVR'] = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': (0.01, 1000.0, 'log-uniform'),
        'gamma': ['scale', 'auto'] + [(0.0001, 1.0, 'log-uniform')]
    }
    
    param_ranges['KNN'] = {
        'n_neighbors': (1, 30),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    # Neural Network model
    param_ranges['MLP'] = {
        'hidden_layer_sizes': [1, 2, 3, 4],  # Will map to actual tuples later
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': (0.00001, 0.1, 'log-uniform'),
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Gaussian Process Regression
    param_ranges['GPR'] = {
        'kernel': [1, 2, 3],  # Will map to actual kernels later
        'alpha': (1e-12, 1e-4, 'log-uniform'),
        'normalize_y': [True, False],
        'n_restarts_optimizer': (0, 5)
    }
    
    return param_grids, param_ranges
