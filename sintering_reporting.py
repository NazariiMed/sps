import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

def save_model_artifacts(model, scaler, feature_names, directory='models'):
    """
    Saves the trained model, scaler, and feature names for later use.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    joblib.dump(model, os.path.join(directory, 'sintering_model_latest.joblib'))
    joblib.dump(scaler, os.path.join(directory, 'sintering_scaler_latest.joblib'))
    joblib.dump(feature_names, os.path.join(directory, 'feature_names.joblib'))
    
    print(f"\n[INFO] Model artifacts saved to '{directory}/'")

def generate_pdf_report(y_true, all_predictions, results, feature_names, best_model, directory='reports'):
    """
    Generates a comprehensive PDF report with plots and metrics for ALL top models.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"sintering_report_{timestamp}.pdf")
    
    print(f"\n[INFO] Generating PDF report: {filename}")
    
    with PdfPages(filename) as pdf:
        # --- Page 1: Summary Table ---
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        
        text = "SPS Sintering Regression Analysis Report\n"
        text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        text += "="*60 + "\n\n"
        
        text += "Model Performance (Validation Data):\n"
        text += "-"*40 + "\n"
        
        max_len = max([len(r['model']) for r in results])
        header = f"{'Model':<{max_len}} | {'R2':<8} | {'RMSE':<8} | {'MAE':<8}\n"
        text += header
        text += "-"*len(header) + "\n"
        
        for res in results:
            text += f"{res['model']:<{max_len}} | {res['r2']:.4f}   | {res['rmse']:.4f}   | {res['mae']:.4f}\n"
            
        text += "\n" + "="*60 + "\n"
        text += "Best Model Selected: " + results[0]['model']
        
        ax.text(0.1, 0.9, text, fontsize=11, va='top', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # --- Page 2: Feature Importance (Best Model) ---
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            else:
                importances = np.abs(best_model.coef_)
                
            indices = np.argsort(importances)[::-1][:20]
            
            ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.invert_yaxis()
            ax.set_xlabel('Relative Importance')
            ax.set_title(f'Top 20 Feature Importance - {results[0]["model"]}')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
        # --- Page 3: All Models Time Series Comparison ---
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(y_true, label='Actual Data', color='black', linewidth=2.5, alpha=0.5)
        
        # Plot top 5 models
        top_models = [r['model'] for r in results[:5]]
        colors = plt.cm.tab10.colors
        
        for i, model_name in enumerate(top_models):
            if model_name in all_predictions:
                ax.plot(all_predictions[model_name], label=f'{model_name}', 
                        color=colors[i%len(colors)], linewidth=1.5, alpha=0.7)
                
        ax.set_title('Top 5 Models Comparison (Time Series)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Rel. Piston Trav')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # --- Pages 4+: Individual Model Analysis (Top 5) ---
        for i, model_res in enumerate(results[:5]):
            model_name = model_res['model']
            if model_name not in all_predictions:
                continue
                
            y_pred = all_predictions[model_name]
            residuals = y_true - y_pred
            
            # Create a 2x2 grid for detailed analysis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Detailed Analysis: {model_name}", fontsize=16)
            
            # 1. Actual vs Predicted Scatter
            axes[0, 0].scatter(y_true, y_pred, alpha=0.5, color='blue')
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals vs Predicted
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predicted')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Residuals Histogram
            axes[1, 0].hist(residuals, bins=50, color='purple', alpha=0.7)
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Residual Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            
            # 4. Zoomed Time Series (First 200 points)
            zoom_len = min(200, len(y_true))
            axes[1, 1].plot(range(zoom_len), y_true[:zoom_len], label='Actual', color='black')
            axes[1, 1].plot(range(zoom_len), y_pred[:zoom_len], label='Predicted', color='red', linestyle='--')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title(f'Zoomed View (First {zoom_len} points)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

    print(f"[INFO] Report generated successfully.")