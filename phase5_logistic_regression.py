# Phase 5: Logistic Regression for Fake Job Detection
# Binary classification model saved to fake_detector.pkl

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from phase3_preprocessing import get_preprocessed_data

def train_fake_detector():
    print("--- Phase 5: Logistic Regression (Fake Job Classifier) ---")
    data_path = os.path.join("dataset", "jobs.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("Loading data...")
    X_train, X_test, _, _, y_train_fake, y_test_fake, _ = get_preprocessed_data(data_path)
    
    # Train Logistic Regression Model
    print("Training classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train_fake)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_test_fake, y_pred)
    prec = precision_score(y_test_fake, y_pred, zero_division=0)
    rec = recall_score(y_test_fake, y_pred, zero_division=0)
    f1 = f1_score(y_test_fake, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_fake, y_pred)
    
    print("\n--- Model Performance ---")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  TN = {cm[0][0]} | FP = {cm[0][1]}")
    print(f"  FN = {cm[1][0]} | TP = {cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_fake, y_pred, zero_division=0))
    
    # Save model
    model_name = "fake_detector.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(classifier, f)
    print(f"Model saved to '{model_name}'")
    
    # Plot Confusion Matrix using Matplotlib
    plot_matrix(cm)

def plot_matrix(cm):
    os.makedirs("graphs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    labels = ['Genuine (0)', 'Fake (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14)
            
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Fake Job Detection', fontsize=12, fontweight='bold')
    
    out_file = os.path.join("graphs", "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    print(f"Confusion matrix image saved to '{out_file}'")

if __name__ == '__main__':
    train_fake_detector()
