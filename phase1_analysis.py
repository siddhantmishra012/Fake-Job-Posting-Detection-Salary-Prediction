# Phase 1: Data Analysis & Exploration
# Submitted for B.Tech CSE Assignment - Fake Job Posting Detection

import pandas as pd
import os

def load_dataset(csv_path):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(csv_path):
        print(f"[Error] Could not find dataset at: {csv_path}")
        return None
    
    print(f"Reading dataset: {csv_path}...")
    return pd.read_csv(csv_path)

def analyze_data(df):
    """Displays dataset info, shapes, missing values, and statistics."""
    print("\n--- DATASET OVERVIEW ---")
    print("Shape (Rows, Cols):", df.shape)
    
    print("\nColumns in Dataset:")
    print(list(df.columns))
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    print("\nTotal Duplicate Rows:", df.duplicated().sum())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nFirst 10 Rows:")
    print(df.head(10))
    
    print("\nLast 10 Rows:")
    print(df.tail(10))

def print_key_insights(df):
    """Calculates and prints dataset insights for fake jobs and salaries."""
    print("\n--- KEY INSIGHTS & STATS ---")
    
    total = len(df)
    fake_cnt = df['Fake'].sum()
    real_cnt = total - fake_cnt
    
    print(f"Genuine Jobs: {real_cnt} ({(real_cnt/total)*100:.2f}%)")
    print(f"Fake Jobs:    {fake_cnt} ({(fake_cnt/total)*100:.2f}%)")
    
    avg_sal = df['Salary'].mean()
    max_sal = df['Salary'].max()
    min_sal = df['Salary'].min()
    
    print(f"\nAverage Salary: Rs. {avg_sal:,.2f}")
    print(f"Highest Salary: Rs. {max_sal:,}")
    print(f"Lowest Salary:  Rs. {min_sal:,}")
    
    print("\nAverage Salary by Company Size:")
    size_sal = df.groupby('Company_Size')['Salary'].mean()
    for size, sal in size_sal.items():
        print(f"  {size:<12}: Rs. {sal:,.2f}")
        
    print("\nAverage Salary by Education Degree:")
    deg_sal = df.groupby('Degree')['Salary'].mean()
    for deg, sal in deg_sal.items():
        print(f"  {deg:<12}: Rs. {sal:,.2f}")
        
    print("\nAverage Salary by Experience Level:")
    exp_sal = df.groupby('Experience')['Salary'].mean()
    for exp, sal in exp_sal.items():
        print(f"  {exp:<12}: Rs. {sal:,.2f}")
        
    print("\nFake Jobs Count by Company Size:")
    print(df[df['Fake'] == 1]['Company_Size'].value_counts().to_string())
    
    print("\nTop 5 Email Domains in Fake Job Postings:")
    print(df[df['Fake'] == 1]['Email_Domain'].value_counts().head(5).to_string())
    
    fake_fee = df[(df['Fake'] == 1) & (df['Application_Fee'] == 1)].shape[0]
    print(f"\nFake Jobs asking for Application Fee: {fake_fee}")
    
    print("\nTop 10 Highest Paying Jobs:")
    top10 = df.nlargest(10, 'Salary')[['Job_Title', 'Company', 'Salary']]
    for idx, row in top10.iterrows():
        print(f"  - {row['Job_Title']} @ {row['Company']}: Rs. {row['Salary']:,}")

if __name__ == '__main__':
    data_path = os.path.join("dataset", "jobs.csv")
    df = load_dataset(data_path)
    if df is not None:
        analyze_data(df)
        print_key_insights(df)
        print("\n[Done] Phase 1 Analysis completed successfully.")
