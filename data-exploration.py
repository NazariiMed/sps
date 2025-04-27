import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# Define file paths
file_paths = [
    '160508-1021-1000,0min,56kN.csv',
    '160508-1022-900,0min,56kN.csv',
    '200508-1023-1350,0min,56kN.csv',
    '200508-1024-1200,0min,56kN.csv'
]


def load_and_explore_data(file_paths):
    """
    Load all CSV files and perform exploratory data analysis.

    Args:
        file_paths: List of CSV file paths
    """
    all_data = []

    print("Loading and exploring data files...")

    for i, file_path in enumerate(file_paths):
        print(f"\nFile {i + 1}: {file_path}")

        # Read the CSV file with proper settings for European number format
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',', header=0)
            # Add a file identifier column
            df['file_id'] = i
            all_data.append(df)

            # Display basic information
            print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            print("  First few rows:")
            print(df.head(3).to_string())

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print("\n  Missing values:")
                print(missing_values[missing_values > 0])

            # Analyze target variable
            target_col = 'Rel. Piston Trav'
            if target_col in df.columns:
                print(f"\n  {target_col} statistics:")
                print(f"    Min: {df[target_col].min()}")
                print(f"    Max: {df[target_col].max()}")
                print(f"    Mean: {df[target_col].mean():.4f}")
                print(f"    Std Dev: {df[target_col].std():.4f}")
                print(f"    Unique values: {df[target_col].nunique()}")

                # Check for precision issues
                decimal_places = df[target_col].astype(str).str.split('.').str[1].str.len().max()
                print(f"    Decimal places: {decimal_places}")

            # Quick correlation analysis
            if target_col in df.columns:
                # Get correlations with target
                corr = df.corr()[target_col].sort_values(ascending=False)
                print("\n  Top 5 correlations with target:")
                print(corr.head(6).to_string())  # +1 to include the target itself
                print("\n  Bottom 5 correlations with target:")
                print(corr.tail(5).to_string())

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Combine all data for overall analysis
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print("\nCombined dataset:")
        print(f"  Total rows: {combined_df.shape[0]}, Columns: {combined_df.shape[1]}")

        return combined_df

    return None


def plot_target_variable(df, target_col='Rel. Piston Trav'):
    """
    Create visualizations for the target variable.

    Args:
        df: DataFrame with all data
        target_col: Name of target column
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return

    print(f"\nGenerating plots for {target_col}...")

    # Create a copy of the dataframe with only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])

    # Set up figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Distribution of target variable
    sns.histplot(df[target_col], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(f'Distribution of {target_col}')
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel('Frequency')

    # Plot 2: Target variable by file
    sns.boxplot(x='file_id', y=target_col, data=df, ax=axes[0, 1])
    axes[0, 1].set_title(f'{target_col} by File')
    axes[0, 1].set_xlabel('File ID')
    axes[0, 1].set_ylabel(target_col)

    # Plot 3: Target variable over time (for first 1000 points)
    sample_size = min(1000, df.shape[0])
    axes[1, 0].plot(df['Nr.'].head(sample_size), df[target_col].head(sample_size))
    axes[1, 0].set_title(f'{target_col} Over Time (First {sample_size} Points)')
    axes[1, 0].set_xlabel('Record Number')
    axes[1, 0].set_ylabel(target_col)

    # Plot 4: Correlation heatmap (top correlated features)
    try:
        # Get absolute correlations with target from numeric columns only
        corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        top_features = corr.head(10).index  # Top 10 features

        # Create correlation matrix for selected features
        corr_matrix = numeric_df[top_features].corr()

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Heatmap (Top Features)')

        # Prepare for additional plot with top features
        top_correlated = corr.head(6).index.tolist()
        if target_col in top_correlated:
            top_correlated.remove(target_col)  # Exclude target itself
        top_correlated = top_correlated[:4]  # Get top 4

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        axes[1, 1].set_title('Correlation Heatmap (Error occurred)')
        top_correlated = []

    plt.tight_layout()
    plt.show()  # Add this line to display the plot
    plt.close()

    # Additional plot: Scatter plots of top correlated features vs target
    if top_correlated:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, feature in enumerate(top_correlated):
            if i < 4:  # Plot top 4 correlated features
                sns.scatterplot(x=feature, y=target_col, data=df.sample(min(1000, df.shape[0])),
                                alpha=0.5, ax=axes[i])
                axes[i].set_title(f'{feature} vs {target_col}')

        # Hide unused subplots
        for j in range(len(top_correlated), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('feature_correlations.png')
        plt.show()  # Add this line to display the plot
        plt.close()


def analyze_feature_distributions(df):
    """
    Analyze the distributions of key features.

    Args:
        df: DataFrame with all data
    """
    print("\nAnalyzing feature distributions...")

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove certain columns we don't need to visualize
    cols_to_exclude = ['Nr.', 'file_id', 'Abs. Piston Trav']
    feature_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    # Select top features based on data exploration
    selected_features = [
        'MTC1', 'MTC2', 'MTC3', 'Pyrometer', 'SV Temperature',
        'SV Power', 'SV Force', 'AV Force', 'AV Speed',
        'I RMS', 'U RMS', 'Heating power'
    ]

    # Ensure all selected features exist in the dataframe
    selected_features = [f for f in selected_features if f in df.columns]

    # Create distribution plots for selected features
    n_cols = 3
    n_rows = (len(selected_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        if i < len(axes):
            sns.histplot(df[feature].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)

    # Hide unused subplots
    for j in range(len(selected_features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.show()  # Add this line to display the plot
    plt.close()


def plot_time_series_by_file(df, target_col='Rel. Piston Trav'):
    """
    Plot time series of target variable for each file.

    Args:
        df: DataFrame with all data
        target_col: Name of target column
    """
    print("\nPlotting time series by file...")

    # Create a figure
    plt.figure(figsize=(15, 8))

    # Plot for each file ID
    for file_id in df['file_id'].unique():
        file_data = df[df['file_id'] == file_id]
        plt.plot(range(len(file_data)), file_data[target_col],
                 label=f'File {file_id}', alpha=0.7)

    plt.title(f'{target_col} Time Series by File')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('time_series_by_file.png')
    plt.show()  # Add this line to display the plot
    plt.close()


def main():
    """Main execution function"""
    print("SPS Sintering Data Exploration")

    # Load and explore data
    combined_df = load_and_explore_data(file_paths)

    if combined_df is not None:
        # Generate plots
        plot_target_variable(combined_df)
        analyze_feature_distributions(combined_df)
        plot_time_series_by_file(combined_df)

        print("\nData exploration complete. Plots saved.")


if __name__ == "__main__":
    main()