# Phase 2: Data Visualization
# Matplotlib graphs saved to graphs/ folder

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def setup_graphs_folder(folder="graphs"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

def save_and_display(fig, filename):
    out_path = os.path.join("graphs", filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot -> {out_path}")
    plt.show()

# 1. Bar chart: Fake vs Genuine
def plot_fake_vs_genuine(df):
    counts = df['Fake'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(['Genuine (0)', 'Fake (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#2b5c8f', '#d9534f'])
    ax.set_title('Fake vs Genuine Job Postings', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Jobs')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '1_fake_vs_genuine.png')

# 2. Histogram: Salary Distribution
def plot_salary_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['Salary'].dropna(), bins=30, color='#337ab7', edgecolor='white')
    ax.set_title('Salary Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Salary (Rs.)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '2_salary_histogram.png')

# 3. Pie chart: Company Size Distribution
def plot_company_size(df):
    counts = df['Company_Size'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#5bc0de', '#5cb85c', '#f0ad4e', '#d9534f'])
    ax.set_title('Company Size Distribution', fontsize=13, fontweight='bold')
    save_and_display(fig, '3_company_size_distribution.png')

# 4. Bar chart: Remote vs Onsite Jobs
def plot_remote_jobs(df):
    counts = df['Remote'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(['Onsite (0)', 'Remote (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#428bca', '#5cb85c'])
    ax.set_title('Remote vs Onsite Jobs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Job Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '4_remote_vs_onsite.png')

# 5. Bar chart: Required Education Degree
def plot_degree_distribution(df):
    counts = df['Degree'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color='#8a6d3b')
    ax.set_title('Distribution of Required Degrees', fontsize=13, fontweight='bold')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '5_degree_distribution.png')

# 6. Scatter plot: Experience vs Salary
def plot_exp_vs_salary(df):
    clean_df = df.dropna(subset=['Experience', 'Salary'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(clean_df['Experience'], clean_df['Salary'], alpha=0.4, color='#31708f')
    ax.set_title('Experience vs Salary', fontsize=13, fontweight='bold')
    ax.set_xlabel('Experience Level')
    ax.set_ylabel('Salary (Rs.)')
    ax.grid(True, linestyle='--', alpha=0.5)
    save_and_display(fig, '6_experience_vs_salary.png')

# 7. Horizontal Bar Chart: Avg Salary by Company Size
def plot_avg_salary_company_size(df):
    avg_sal = df.groupby('Company_Size')['Salary'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(avg_sal.index, avg_sal.values, color='#f0ad4e')
    ax.set_title('Average Salary by Company Size', fontsize=13, fontweight='bold')
    ax.set_xlabel('Average Salary (Rs.)')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    save_and_display(fig, '7_avg_salary_by_company_size.png')

# 8. Bar chart: Fake Jobs by Industry
def plot_fake_jobs_industry(df):
    fake_df = df[df['Fake'] == 1]
    counts = fake_df['Industry'].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color='#c9302c')
    ax.set_title('Fake Jobs Distribution by Industry', fontsize=13, fontweight='bold')
    ax.set_xlabel('Industry')
    ax.set_ylabel('Fake Job Count')
    plt.xticks(rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '8_fake_jobs_by_industry.png')

# 9. Bar chart: Job Type Distribution
def plot_job_types(df):
    counts = df['Job_Type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color='#31b0d5')
    ax.set_title('Job Type Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Job Type')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_display(fig, '9_job_type_distribution.png')

# 10. Boxplot: Salary Outliers
def plot_salary_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(df['Salary'].dropna(), vert=False, patch_artist=True,
               boxprops=dict(facecolor='#d9534f', color='black'),
               medianprops=dict(color='white', linewidth=2))
    ax.set_title('Salary Box Plot (Outlier Analysis)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Salary (Rs.)')
    ax.set_yticklabels(['Salary'])
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    save_and_display(fig, '10_salary_boxplot.png')

def main():
    plt.style.use('ggplot')
    data_file = os.path.join("dataset", "jobs.csv")
    df = load_data(data_file)
    
    if df is not None:
        setup_graphs_folder()
        print("Creating plots...")
        plot_fake_vs_genuine(df)
        plot_salary_histogram(df)
        plot_company_size(df)
        plot_remote_jobs(df)
        plot_degree_distribution(df)
        plot_exp_vs_salary(df)
        plot_avg_salary_company_size(df)
        plot_fake_jobs_industry(df)
        plot_job_types(df)
        plot_salary_boxplot(df)
        print("\n[Done] All 10 plots saved inside graphs/ folder.")

if __name__ == '__main__':
    main()
