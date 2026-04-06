import pandas as pd
import os

def load_overdose_data(data_path="data/VSRR_Provisional_Drug_Overdose_Death_Counts_20260404.csv"):
    if not os.path.exists(data_path):
        return None
    try:
        # Load structured dataset
        df = pd.read_csv(data_path)
        
        # Filter for national-level estimates to extract broad temporal trends
        national_df = df[df['State'] == 'US'].copy()
        
        # Clean numeric data value (strip commas if any, coerce to float)
        national_df['Data Value'] = pd.to_numeric(national_df['Data Value'].astype(str).str.replace(',', ''), errors='coerce')
        
        return national_df
    except Exception as e:
        print(f"Error loading temporal data: {e}")
        return None

def get_drug_trends(query, df=None):
    if df is None:
        df = load_overdose_data()
        if df is None:
            return "Note: Temporal quantitative data is currently unavailable."
            
    query_lower = query.lower()
    targets = []
    
    # Keyword routing to Indicators
    if "heroin" in query_lower: targets.append("Heroin (T40.1)")
    if "cocaine" in query_lower: targets.append("Cocaine (T40.5)")
    if "methadone" in query_lower: targets.append("Methadone (T40.3)")
    if "opioid" in query_lower or "fentanyl" in query_lower: 
        targets.append("Synthetic opioids, excl. methadone (T40.4)")
        
    # Default indicator if no specific substance mapped
    if not targets:
        targets.append("Number of Drug Overdose Deaths")

    results = []
    for t in targets:
        sub_df = df[(df['Indicator'] == t) & (df['Year'].notna())]
        if sub_df.empty:
            continue
            
        # Group by year safely handling maximums to get general total trajectory
        yearly = sub_df.groupby('Year')['Data Value'].max().dropna().reset_index()
        yearly = yearly.sort_values(by='Year')
        
        if len(yearly) >= 2:
            latest = yearly.iloc[-1]
            prev = yearly.iloc[-2]
            
            if prev['Data Value'] > 0:
                pct_change = ((latest['Data Value'] - prev['Data Value']) / prev['Data Value']) * 100
                trend = "increased" if pct_change > 0 else "decreased"
                
                results.append(f"CDC stats show national deaths for '{t}' {trend} by {abs(pct_change):.1f}% from {int(prev['Year'])} to {int(latest['Year'])}, reaching {int(latest['Data Value'])} estimated deaths.")
            else:
                results.append(f"CDC stats report {int(latest['Data Value'])} provisional deaths for '{t}' in {int(latest['Year'])}.")
        elif len(yearly) == 1:
            results.append(f"CDC stats report {int(yearly.iloc[0]['Data Value'])} provisional deaths for '{t}' in {int(yearly.iloc[0]['Year'])}.")
            
    if results:
        return " | ".join(results)
    else:
        return "No specific national CDC temporal trend identified for this explicit query parameters."
