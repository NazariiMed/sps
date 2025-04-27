import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
file_paths = [
    '160508-1021-1000,0min,56kN.csv',
    '160508-1022-900,0min,56kN.csv',
    '200508-1023-1350,0min,56kN.csv',
    '200508-1024-1200,0min,56kN.csv'
]

# Target column to smooth
TARGET_COLUMN = 'Rel. Piston Trav'

def load_file(file_path):
    """
    Load a CSV file with European number format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(file_path, sep=';', decimal=',', header=0)
        print(f"Loaded {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_target_column(df, target_col):
    """
    Analyze the target column to understand precision issues.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        
    Returns:
        Dictionary with analysis results
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return None
        
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
    
    print(f"\nAnalysis of '{target_col}':")
    print(f"  Unique values: {results['unique_values']} out of {results['total_values']} total values")
    print(f"  Minimum non-zero difference: {results['min_nonzero_diff']:.8f}")
    print(f"  Zero differences: {results['zero_diff_count']} ({results['zero_diff_percentage']:.2f}% of all consecutive pairs)")
    print(f"  Maximum consecutive repeated values: {results['max_consecutive_repeats']}")
    
    return results

def smooth_target_column(df, target_col, method='noise', params=None):
    """
    Smooth the target column to address precision issues.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        method: Smoothing method to use ('noise', 'spline', or 'rolling')
        params: Parameters for the smoothing method
        
    Returns:
        DataFrame with the smoothed target column
    """
    # Make a copy to avoid modifying the original
    smoothed_df = df.copy()
    
    if target_col not in smoothed_df.columns:
        print(f"Target column '{target_col}' not found in data")
        return smoothed_df
    
    # Extract target column
    target_values = smoothed_df[target_col].values
    
    if method == 'noise':
        # Default parameters
        if params is None:
            params = {'noise_scale': 0.0001}
        
        # Add small noise to break plateaus
        noise_scale = params.get('noise_scale', 0.0001)
        np.random.seed(42)  # For reproducibility
        smoothed_values = target_values + np.random.normal(0, noise_scale, len(target_values))
        
    elif method == 'spline':
        from scipy.interpolate import UnivariateSpline
        
        # Default parameters
        if params is None:
            params = {'s': 0.01}
        
        # Use spline interpolation
        x = np.arange(len(target_values))
        s = params.get('s', 0.01)  # Smoothing factor
        spline = UnivariateSpline(x, target_values, s=s)
        smoothed_values = spline(x)
        
    elif method == 'rolling':
        # Default parameters
        if params is None:
            params = {'window': 3, 'center': True}
        
        # Use rolling average
        window = params.get('window', 3)
        center = params.get('center', True)
        smoothed_series = pd.Series(target_values).rolling(
            window=window, center=center, min_periods=1).mean()
        smoothed_values = smoothed_series.values
        
    else:
        print(f"Unknown smoothing method: {method}")
        return smoothed_df
    
    # Update the target column in the DataFrame
    smoothed_df[target_col] = smoothed_values
    
    return smoothed_df

def plot_comparison(original_df, smoothed_df, target_col, file_name=None, samples=1000):
    """
    Plot comparison between original and smoothed data.
    
    Args:
        original_df: DataFrame with original data
        smoothed_df: DataFrame with smoothed data
        target_col: Name of the target column
        file_name: Name of the file (for title)
        samples: Number of samples to plot
    """
    if target_col not in original_df.columns or target_col not in smoothed_df.columns:
        print(f"Target column '{target_col}' not found in data")
        return
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Get data for plotting
    original_values = original_df[target_col].values[:samples]
    smoothed_values = smoothed_df[target_col].values[:samples]
    x = np.arange(len(original_values))
    
    # Plot 1: Overview
    axes[0].plot(x, original_values, label='Original', alpha=0.7)
    axes[0].plot(x, smoothed_values, label='Smoothed', alpha=0.7)
    axes[0].set_title(f"Overview of {target_col}" + (f" ({file_name})" if file_name else ""))
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel(target_col)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed section (first 200 points)
    zoom_end = min(200, len(original_values))
    axes[1].plot(x[:zoom_end], original_values[:zoom_end], label='Original', alpha=0.7)
    axes[1].plot(x[:zoom_end], smoothed_values[:zoom_end], label='Smoothed', alpha=0.7)
    axes[1].set_title(f"Zoomed View (First {zoom_end} Points)")
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel(target_col)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Difference between original and smoothed
    diff = smoothed_values - original_values
    axes[2].plot(x, diff, label='Smoothed - Original', color='green', alpha=0.7)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title('Difference (Smoothed - Original)')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_smoothed_file(df, original_path, suffix="_smoothed"):
    """
    Save the DataFrame to a new CSV file with European number format.
    
    Args:
        df: DataFrame to save
        original_path: Path to the original CSV file
        suffix: Suffix to add to the new filename
        
    Returns:
        Path to the saved file
    """
    # Create new filename
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}{suffix}{ext}"
    
    # Save with European number format
    df.to_csv(new_path, sep=';', decimal=',', index=False)
    print(f"Saved smoothed data to {new_path}")
    
    return new_path

def process_file(file_path, smoothing_method, params=None):
    """
    Process a single file: load, analyze, smooth, plot comparison, and save.
    
    Args:
        file_path: Path to the CSV file
        smoothing_method: Method to use for smoothing
        params: Parameters for the smoothing method
        
    Returns:
        Path to the saved smoothed file
    """
    # Load the file
    df = load_file(file_path)
    if df is None:
        return None
    
    # Analyze the target column
    analysis = analyze_target_column(df, TARGET_COLUMN)
    if analysis is None:
        return None
    
    # Adjust smoothing parameters based on analysis if not provided
    if params is None:
        if smoothing_method == 'noise':
            # Use 1/10 of the minimum non-zero difference
            noise_scale = max(0.00001, abs(analysis['min_abs_nonzero_diff']) / 10)
            params = {'noise_scale': noise_scale}
            print(f"Using noise scale: {noise_scale:.8f}")
        elif smoothing_method == 'spline':
            # Adjust smoothing factor based on data range
            data_range = df[TARGET_COLUMN].max() - df[TARGET_COLUMN].min()
            s = 0.0001 * data_range * len(df)
            params = {'s': s}
            print(f"Using spline smoothing factor: {s:.8f}")
        elif smoothing_method == 'rolling':
            # Use window size based on average run length of repeated values
            window = max(3, int(analysis['avg_consecutive_repeats'] / 2))
            params = {'window': window, 'center': True}
            print(f"Using rolling window size: {window}")
    
    # Smooth the target column
    smoothed_df = smooth_target_column(df, TARGET_COLUMN, smoothing_method, params)
    
    # Plot comparison
    plot_comparison(df, smoothed_df, TARGET_COLUMN, os.path.basename(file_path))
    
    # Save the smoothed data
    smoothed_path = save_smoothed_file(smoothed_df, file_path)
    
    return smoothed_path

def main():
    """Main execution function"""
    print("SPS Data Smoothing Utility")
    print("==========================")
    
    # Smoothing parameters
    smoothing_method = 'noise'  # 'noise', 'spline', or 'rolling'
    
    # Process each file
    smoothed_files = []
    for file_path in file_paths:
        print(f"\nProcessing {file_path}...")
        smoothed_path = process_file(file_path, smoothing_method)
        if smoothed_path:
            smoothed_files.append(smoothed_path)
    
    print("\nProcessing complete!")
    print(f"Created {len(smoothed_files)} smoothed files:")
    for file_path in smoothed_files:
        print(f"  {file_path}")

if __name__ == "__main__":
    main()