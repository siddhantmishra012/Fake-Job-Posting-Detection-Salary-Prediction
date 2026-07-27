"""
Generate Synthetic Dataset: jobs.csv
=====================================
This script generates a realistic synthetic dataset for the
Fake Job Posting Detection & Salary Prediction project.

The dataset contains 2000 job postings with columns that support
both classification (Fake detection) and regression (Salary prediction).
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_ROWS = 2000

# --- Define possible values for each column ---

job_titles = [
    "Software Engineer", "Data Analyst", "Marketing Manager", "HR Executive",
    "Product Manager", "Web Developer", "Data Scientist", "Business Analyst",
    "Project Manager", "UX Designer", "DevOps Engineer", "Content Writer",
    "Sales Executive", "Financial Analyst", "Operations Manager",
    "Machine Learning Engineer", "Graphic Designer", "Quality Analyst",
    "Network Administrator", "Database Administrator", "Cybersecurity Analyst",
    "Cloud Engineer", "Mobile Developer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "System Administrator", "Technical Writer",
    "SEO Specialist", "Digital Marketing Executive"
]

company_names = [
    "TechCorp", "Innovate Solutions", "DataMinds", "CloudNine Systems",
    "NextGen Technologies", "FutureTech", "Zenith Labs", "PrimeSoft",
    "CoreLogic", "BlueChip Analytics", "Synergy Tech", "AlphaWave",
    "BrightPath Solutions", "QuantumLeap", "ApexDigital", "NovaTech",
    "Vertex Systems", "PeakPerformance IT", "EliteCode", "SmartBridge"
]

company_sizes = ["Small", "Medium", "Large", "Enterprise"]
company_size_weights = [0.25, 0.35, 0.25, 0.15]

experience_levels = ["Entry Level", "Mid Level", "Senior", "Executive", "Internship"]
experience_weights = [0.30, 0.30, 0.20, 0.10, 0.10]

degrees = ["Bachelor's", "Master's", "PhD", "Diploma", "High School"]
degree_weights = [0.40, 0.25, 0.10, 0.15, 0.10]

job_types = ["Full-Time", "Part-Time", "Contract", "Freelance", "Internship"]
job_type_weights = [0.45, 0.15, 0.15, 0.10, 0.15]

industries = [
    "Information Technology", "Finance", "Healthcare", "Education",
    "Manufacturing", "Retail", "Consulting", "Telecommunications",
    "Media", "Automotive"
]

skills_list = [
    "Python", "Java", "SQL", "Excel", "Communication", "Leadership",
    "Machine Learning", "Cloud Computing", "Data Analysis", "Project Management",
    "JavaScript", "C++", "Marketing", "Accounting", "Design"
]

email_domains = [
    "company.com", "techcorp.com", "innovate.com", "hr-solutions.com",
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "jobs-apply.com", "hire-now.net"
]

locations = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Noida",
    "Gurgaon", "Chandigarh", "Lucknow", "Indore", "Kochi"
]


def generate_salary(experience, degree, company_size):
    """Generate a realistic salary based on experience, degree, and company size."""
    base = {
        "Internship": 150000, "Entry Level": 300000,
        "Mid Level": 600000, "Senior": 1000000, "Executive": 1500000
    }
    degree_mult = {
        "High School": 0.7, "Diploma": 0.85,
        "Bachelor's": 1.0, "Master's": 1.2, "PhD": 1.4
    }
    size_mult = {
        "Small": 0.8, "Medium": 1.0, "Large": 1.2, "Enterprise": 1.4
    }
    salary = base[experience] * degree_mult[degree] * size_mult[company_size]
    # Add random noise (+/- 20%)
    noise = np.random.uniform(0.8, 1.2)
    return int(salary * noise)


def generate_dataset():
    """Generate the complete synthetic dataset."""
    data = []

    for i in range(NUM_ROWS):
        # Basic job info
        title = np.random.choice(job_titles)
        company = np.random.choice(company_names)
        location = np.random.choice(locations)
        company_size = np.random.choice(company_sizes, p=company_size_weights)
        experience = np.random.choice(experience_levels, p=experience_weights)
        degree = np.random.choice(degrees, p=degree_weights)
        job_type = np.random.choice(job_types, p=job_type_weights)
        industry = np.random.choice(industries)
        remote = np.random.choice([0, 1], p=[0.6, 0.4])

        # Skills (1-4 random skills)
        num_skills = np.random.randint(1, 5)
        selected_skills = np.random.choice(skills_list, size=num_skills, replace=False)
        skills = ", ".join(selected_skills)

        # Salary
        salary = generate_salary(experience, degree, company_size)

        # Company credibility features
        has_company_website = np.random.choice([0, 1], p=[0.15, 0.85])
        has_company_logo = np.random.choice([0, 1], p=[0.2, 0.8])
        email_domain = np.random.choice(email_domains)
        application_fee = np.random.choice([0, 1], p=[0.85, 0.15])
        urgent_hiring = np.random.choice([0, 1], p=[0.75, 0.25])

        # Determine Fake/Genuine based on realistic rules
        fake_score = 0
        # No website increases fake probability
        if has_company_website == 0:
            fake_score += 2
        # No logo increases fake probability
        if has_company_logo == 0:
            fake_score += 1
        # Free email domains are suspicious
        if email_domain in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]:
            fake_score += 2
        # Suspicious domains
        if email_domain in ["jobs-apply.com", "hire-now.net"]:
            fake_score += 3
        # Application fee is a major red flag
        if application_fee == 1:
            fake_score += 3
        # Urgent hiring is slightly suspicious
        if urgent_hiring == 1:
            fake_score += 1
        # Very high salary for entry level is suspicious
        if experience in ["Entry Level", "Internship"] and salary > 800000:
            fake_score += 2

        # Add randomness to fake determination
        fake_score += np.random.randint(-2, 3)

        # Threshold: score >= 5 means Fake
        fake = 1 if fake_score >= 5 else 0

        # If fake, sometimes inflate salary unrealistically
        if fake == 1 and np.random.random() > 0.5:
            salary = int(salary * np.random.uniform(1.5, 3.0))

        data.append({
            "Job_Title": title,
            "Company": company,
            "Location": location,
            "Company_Size": company_size,
            "Experience": experience,
            "Degree": degree,
            "Job_Type": job_type,
            "Industry": industry,
            "Remote": remote,
            "Skills": skills,
            "Salary": salary,
            "Has_Company_Website": has_company_website,
            "Has_Company_Logo": has_company_logo,
            "Email_Domain": email_domain,
            "Application_Fee": application_fee,
            "Urgent_Hiring": urgent_hiring,
            "Fake": fake
        })

    df = pd.DataFrame(data)

    # Introduce some missing values (~3% random NaN)
    for col in ["Company_Size", "Experience", "Degree", "Industry", "Skills"]:
        mask = np.random.random(NUM_ROWS) < 0.03
        df.loc[mask, col] = np.nan

    # Introduce a few duplicate rows (~1%)
    num_duplicates = int(NUM_ROWS * 0.01)
    duplicate_indices = np.random.choice(NUM_ROWS, size=num_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    # Create dataset directory
    os.makedirs("dataset", exist_ok=True)

    # Generate and save
    df = generate_dataset()
    df.to_csv("dataset/jobs.csv", index=False)

    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Fake jobs: {df['Fake'].sum()} ({df['Fake'].mean()*100:.1f}%)")
    print(f"Genuine jobs: {(df['Fake']==0).sum()} ({(df['Fake']==0).mean()*100:.1f}%)")
    print(f"Salary range: Rs.{df['Salary'].min():,} - Rs.{df['Salary'].max():,}")
    print(f"\nSaved to: dataset/jobs.csv")
