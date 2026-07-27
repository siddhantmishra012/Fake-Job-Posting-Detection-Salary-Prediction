# Phase 4: Linear Regression for Salary Prediction
# Trains LinearRegression model, evaluates error metrics, saves salary_model.pkl

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from phase3_preprocessing import get_preprocessed_data

def train_salary_model():
    print("--- Phase 4: Linear Regression (Salary Model) ---")
    data_path = os.path.join("dataset", "jobs.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("Loading data...")
    X_train, X_test, y_train_sal, y_test_sal, _, _, _ = get_preprocessed_data(data_path)
    
    # Train Linear Regression Model
    print("Training model...")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train_sal)
    
    # Predict on test data
    y_pred = regressor.predict(X_test)
    
    # Evaluate Performance
    mae = mean_absolute_error(y_test_sal, y_pred)
    mse = mean_squared_error(y_test_sal, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_sal, y_pred)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE) : Rs. {mae:.2f}")
    print(f"Mean Squared Error (MSE)  : {mse:.2f}")
    print(f"Root Mean Sq Error (RMSE) : Rs. {rmse:.2f}")
    print(f"R2 Score                  : {r2:.4f}")
    
    # Save trained model
    model_file = "salary_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(regressor, f)
    print(f"\nSaved model to '{model_file}'")
    
    # Print sample comparison table
    print("\nSample Comparisons (Actual vs Predicted):")
    results = pd.DataFrame({
        'Actual Salary (Rs.)': y_test_sal[:15].values,
        'Predicted Salary (Rs.)': y_pred[:15].round(0).astype(int)
    })
    print(results.to_string())
    
    # Plot Actual vs Predicted values
    os.makedirs("graphs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test_sal, y_pred, alpha=0.5, color='#2b5c8f', label='Predicted')
    
    min_v = min(y_test_sal.min(), y_pred.min())
    max_v = max(y_test_sal.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Fit')
    
    ax.set_title('Actual vs Predicted Salary', fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Salary (Rs.)')
    ax.set_ylabel('Predicted Salary (Rs.)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plot_file = os.path.join("graphs", "actual_vs_predicted_salary.png")
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{plot_file}'")

if __name__ == '__main__':
    train_salary_model()
