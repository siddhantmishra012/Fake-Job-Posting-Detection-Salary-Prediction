# Phase 6: Fast Interactive Terminal Menu
# Real-time salary prediction & fake job detection system

import os
import pickle
import warnings
import numpy as np
import pandas as pd

# Suppress all sklearn feature warnings globally
warnings.filterwarnings('ignore')

# Load trained pickle files instantly
def load_pkl(filename):
    if not os.path.exists(filename):
        print(f"\n[Warning] '{filename}' not found. Please run training phases first.")
        return None
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Column names matching training features
FEATURE_COLUMNS = [
    'Company_Size', 'Experience', 'Degree', 'Job_Type', 'Industry',
    'Remote', 'Has_Company_Website', 'Has_Company_Logo',
    'Email_Domain', 'Application_Fee', 'Urgent_Hiring'
]

# LabelEncoder index mappings
COMPANY_SIZES = {"Enterprise": 0, "Large": 1, "Medium": 2, "Small": 3}
EXPERIENCE_LEVELS = {"Entry Level": 0, "Executive": 1, "Internship": 2, "Mid Level": 3, "Senior": 4}
DEGREES = {"Bachelor's": 0, "Diploma": 1, "High School": 2, "Master's": 3, "PhD": 4}
JOB_TYPES = {"Contract": 0, "Freelance": 1, "Full-Time": 2, "Internship": 3, "Part-Time": 4}
INDUSTRIES = {
    "Automotive": 0, "Consulting": 1, "Education": 2, "Finance": 3,
    "Healthcare": 4, "Information Technology": 5, "Manufacturing": 6,
    "Media": 7, "Retail": 8, "Telecommunications": 9
}
EMAIL_DOMAINS = {
    "company.com": 0, "gmail.com": 1, "hire-now.net": 2, "hotmail.com": 3,
    "hr-solutions.com": 4, "innovate.com": 5, "jobs-apply.com": 6,
    "outlook.com": 7, "techcorp.com": 8, "yahoo.com": 9
}

FREE_EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "jobs-apply.com", "hire-now.net"]

def select_option(title, options):
    print(f"\nSelect {title}:")
    keys = list(options.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k}")
        
    while True:
        try:
            choice = int(input(f"Enter option (1-{len(keys)}): "))
            if 1 <= choice <= len(keys):
                selected_key = keys[choice - 1]
                return options[selected_key], selected_key
            print("Invalid option number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def ask_yes_no(prompt):
    while True:
        ans = input(f"  {prompt} (yes/no): ").strip().lower()
        if ans in ['yes', 'y']:
            return 1, 'Yes'
        elif ans in ['no', 'n']:
            return 0, 'No'
        print("Enter 'yes' or 'no'.")

def ask_number(prompt):
    while True:
        try:
            val = float(input(f"  {prompt}: "))
            if val >= 0:
                return val
            print("Value cannot be negative.")
        except ValueError:
            print("Please enter a valid number.")

def predict_salary_menu(salary_model):
    print("\n--- SALARY PREDICTION ---")
    if salary_model is None:
        print("[Error] Salary model is not loaded.")
        return
        
    comp_size, comp_label = select_option("Company Size", COMPANY_SIZES)
    exp, exp_label = select_option("Experience Level", EXPERIENCE_LEVELS)
    degree, deg_label = select_option("Degree Required", DEGREES)
    remote, rem_label = ask_yes_no("Is this a Remote job?")
    jtype, jtype_label = select_option("Job Type", JOB_TYPES)
    ind, ind_label = select_option("Industry", INDUSTRIES)
    
    skills = input("\nEnter Required Skills (comma separated): ").strip()
    if not skills:
        skills = "Not specified"
        
    features_array = np.array([[comp_size, exp, degree, jtype, ind, remote, 1, 1, 0, 0, 0]])
    
    pred_salary = int(max(0, salary_model.predict(features_array)[0]))
    
    print("\n----------------------------------")
    print(f"Details: {exp_label} | {deg_label} | {comp_label} | {ind_label}")
    print(f"Skills : {skills}")
    print("----------------------------------")
    print(f"Estimated Salary: Rs. {pred_salary:,}")
    print("----------------------------------")

def evaluate_risk(has_site, has_logo, email_name, fee, urgent, salary):
    trust = 100
    red_flags = []
    
    if not has_site:
        trust -= 25
        red_flags.append("No Company Website")
    if not has_logo:
        trust -= 15
        red_flags.append("No Company Logo")
    if email_name in FREE_EMAILS:
        trust -= 25
        red_flags.append(f"Uses public/suspicious email ({email_name})")
    if fee:
        trust -= 30
        red_flags.append("Application Fee Required")
    if urgent:
        trust -= 10
        red_flags.append("Urgent Hiring Flag")
    if salary > 2000000:
        trust -= 10
        red_flags.append("Suspiciously High Salary")
        
    trust = max(0, min(100, trust))
    risk = 100 - trust
    return trust, risk, red_flags

def detect_fake_menu(fake_model):
    print("\n--- FAKE JOB DETECTION ---")
    if fake_model is None:
        print("[Error] Fake detection model is not loaded.")
        return
        
    site, _ = ask_yes_no("Does company have a Website?")
    logo, _ = ask_yes_no("Does company have a Logo?")
    email_val, email_name = select_option("Email Domain", EMAIL_DOMAINS)
    fee, _ = ask_yes_no("Is there an Application Fee?")
    urgent, _ = ask_yes_no("Is it Urgent Hiring?")
    comp_size, _ = select_option("Company Size", COMPANY_SIZES)
    exp, _ = select_option("Experience Level", EXPERIENCE_LEVELS)
    salary = ask_number("Advertised Salary (Rs.)")
    degree, _ = select_option("Degree Required", DEGREES)
    
    features_array = np.array([[comp_size, exp, degree, 2, 5, 0, site, logo, email_val, fee, urgent]])
    
    prediction = fake_model.predict(features_array)[0]
    probs = fake_model.predict_proba(features_array)[0]
    confidence = probs[prediction] * 100
    
    trust_score, risk_score, red_flags = evaluate_risk(site, logo, email_name, fee, urgent, salary)
    
    print("\n===============================")
    if prediction == 1:
        print("   PREDICTION: FAKE JOB")
    else:
        print("   PREDICTION: GENUINE JOB")
    print("===============================")
    print(f"Confidence Level: {confidence:.2f}%")
    print(f"Company Trust Score: {trust_score}/100")
    print(f"Fraud Risk Score   : {risk_score}/100")
    
    print("\nReasons / Red Flags:")
    if red_flags:
        for rf in red_flags:
            print(f"  - {rf}")
    else:
        print("  - No major red flags detected.")

def main():
    salary_model = load_pkl('salary_model.pkl')
    fake_model = load_pkl('fake_detector.pkl')
    
    while True:
        print("\n===========================")
        print("   FAKE JOB DETECTOR MENU  ")
        print("===========================")
        print("1. Predict Salary")
        print("2. Detect Fake Job")
        print("3. Exit")
        
        c = input("\nEnter choice (1-3): ").strip()
        if c == '1':
            predict_salary_menu(salary_model)
        elif c == '2':
            detect_fake_menu(fake_model)
        elif c == '3':
            print("\nExiting program. Bye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")

if __name__ == '__main__':
    main()
