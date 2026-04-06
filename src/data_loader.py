import os
import sys
import pandas as pd

def process_file(raw_name, clean_name, data_path, n_samples):
    file_path = os.path.join(data_path, raw_name)
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping.")
        return False
        
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Cleaning and anonymizing {raw_name}...")
    df = df.dropna(subset=['review', 'condition'])
    df = df.rename(columns={'review': 'text'})
    df['text'] = df['text'].str.replace(r'&#039;', "'", regex=True)
    df['text'] = df['text'].str.replace(r'<[^>]+>', "", regex=True)
    
    n_actual = min(n_samples, len(df))
    sample_df = df.sample(n=n_actual, random_state=42)
    
    clean_path = os.path.join(data_path, clean_name)
    sample_df.to_csv(clean_path, index=False)
    print(f"Cleaned data saved to {clean_path}. Total sampled records ready: {len(sample_df)}")
    return True

def process_demographics(raw_name, append_to_name, data_path):
    file_path = os.path.join(data_path, raw_name)
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return False
        
    print(f"Loading and stringifying demographic data from {file_path}...")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['ESTIMATE', 'YEAR', 'STUB_LABEL'])
    
    text_list = []
    for _, row in df.iterrows():
        panel = row.get('PANEL', 'Drug overdose')
        label = row.get('STUB_LABEL', 'All persons')
        year = row.get('YEAR', 'Unknown year')
        est = row.get('ESTIMATE', '0')
        unit = row.get('UNIT', 'deaths per 100,000')
        
        # Simplify redundant text
        unit_str = str(unit).replace(" resident population, age-adjusted", "").replace(" resident population, crude", "")
        text = f"CDC Demographic Stat: In {year}, the rate for {panel} among {label} was {est} {unit_str}."
        text_list.append(text)
        
    demo_df = pd.DataFrame({'text': text_list, 'condition': 'Demographic Statistics'})
    
    clean_path = os.path.join(data_path, append_to_name)
    if os.path.exists(clean_path):
        existing_df = pd.read_csv(clean_path)
        combined_df = pd.concat([existing_df, demo_df], ignore_index=True)
        combined_df.to_csv(clean_path, index=False)
        print(f"Appended {len(demo_df)} demographic records to {clean_path}.")
    else:
        demo_df.to_csv(clean_path, index=False)
    return True

def clean_data(data_path="data/"):
    """
    Cleans and anonymizes the raw data for both train and test sets.
    """
    train_success = process_file("drugsComTrain_raw.csv", "cleaned_train_reviews.csv", data_path, 5000)
    test_success = process_file("drugsComTest_raw.csv", "cleaned_test_reviews.csv", data_path, 2000)
    process_demographics("Drug_overdose_death_rates,_by_drug_type,_sex,_age,_race,_and_Hispanic_origin__United_States_20260404.csv", "cleaned_train_reviews.csv", data_path)
    
    if not train_success:
        print("Error: Expected file drugsComTrain_raw.csv not found.")
        sys.exit(1)

if __name__ == "__main__":
    clean_data()
