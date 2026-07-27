# Phase 3: Data Preprocessing
# Cleaning missing values, encoding categories, and splitting data

import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_preprocessed_data(file_path="dataset/jobs.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset missing at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # 1. Remove duplicate rows
    init_count = len(df)
    df = df.drop_duplicates()
    dups_removed = init_count - len(df)
    print(f"Removed {dups_removed} duplicate rows.")
    
    # 2. Fill missing values (median for numbers, mode for categorical strings)
    missing_count = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    print(f"Filled {missing_count} missing values.")
    
    # 3. Label Encoding for categorical columns
    encoders = {}
    cat_columns = ['Company_Size', 'Experience', 'Degree', 'Job_Type', 'Industry', 'Email_Domain']
    
    for col in cat_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Save encoders for interactive prediction phase
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
        
    # 4. Feature Selection & Targets
    drop_cols = ['Job_Title', 'Company', 'Location', 'Skills', 'Salary', 'Fake']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    y_salary = df['Salary']
    y_fake = df['Fake']
    
    # 5. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train_sal, y_test_sal, y_train_fake, y_test_fake = train_test_split(
        X, y_salary, y_fake, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train_sal, y_test_sal, y_train_fake, y_test_fake, encoders

def main():
    print("--- Preprocessing Phase ---")
    data_path = os.path.join("dataset", "jobs.csv")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    X_train, X_test, y_train_sal, y_test_sal, y_train_fake, y_test_fake, encoders = get_preprocessed_data(data_path)
    
    print("\nData Preprocessing Complete:")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("Features used:", list(X_train.columns))
    print("Encoders saved to label_encoders.pkl")

if __name__ == '__main__':
    main()
