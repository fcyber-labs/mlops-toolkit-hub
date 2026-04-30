"""
Advanced feature engineering for LendingClub data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

def engineer_lendingclub_features(df):
    """
    Add advanced features 
    """
    print("\n[Feature Engineering] Adding advanced features...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    #  Handle missing values 
    print("  - Handling missing values...")
    
    # Fill mort_acc using total_acc average 
    if 'total_acc' in df.columns and 'mort_acc' in df.columns:
        total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()
        
        def fill_mort_acc(total_acc, mort_acc):
            if pd.isna(mort_acc):
                return total_acc_avg.get(total_acc, 0)
            return mort_acc
        
        df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
    
    #  Process categorical variables
    print("  - Processing categorical variables...")
    
    # Term to numeric
    if 'term' in df.columns:
        term_map = {' 36 months': 36, ' 60 months': 60}
        if df['term'].dtype == 'object':
            df['term'] = df['term'].map(term_map)
    
    # Process pub_rec, mort_acc, pub_rec_bankruptcies
    def process_binary(number):
        return 0 if number == 0.0 else 1
    
    for col in ['pub_rec', 'pub_rec_bankruptcies']:
        if col in df.columns:
            df[col] = df[col].apply(process_binary)
    
    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].apply(lambda x: 0 if x == 0.0 else (1 if x >= 1.0 else x))
    
    #  Extract zip code 
    print("  - Extracting zip codes...")
    if 'address' in df.columns:
        df['zip_code'] = df['address'].apply(lambda x: str(x)[-5:] if pd.notna(x) else '00000')
    
    #  Extract year from earliest_cr_line
    print("  - Processing credit history...")
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['credit_history_years'] = (pd.Timestamp.now() - df['earliest_cr_line']).dt.days / 365
        df['earliest_cr_line'] = df['earliest_cr_line'].dt.year
    
    # Handle outliers 
    print("  - Capping outliers...")
    outlier_columns = {
        'annual_inc': 250000,
        'dti': 50,
        'open_acc': 40,
        'total_acc': 80,
        'revol_util': 120,
        'revol_bal': 250000
    }
    
    for col, cap in outlier_columns.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    
    # Create derived features
    print("  - Creating derived features...")
    
    # Loan to income ratio
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
        df['loan_to_income'] = df['loan_to_income'].clip(upper=5)
    
    # Interest rate bins 
    if 'int_rate' in df.columns:
        df['int_rate_binned'] = pd.cut(df['int_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    #  Log transform skewed features
    print("  - Log transforming skewed features...")
    skewed_features = ['loan_amnt', 'annual_inc', 'revol_bal']
    for col in skewed_features:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    #  Handle high-cardinality categoricals with frequency encoding
    print("  - Encoding high-cardinality features...")
    high_cardinality = ['zip_code', 'purpose', 'home_ownership']
    for col in high_cardinality:
        if col in df.columns:
            freq_encoding = df[col].value_counts(normalize=True).to_dict()
            df[f'{col}_freq'] = df[col].map(freq_encoding)
    
    print(f" Feature engineering complete! New shape: {df.shape}")
    
    return df

def preprocess_lendingclub_pipeline(df, is_training=True, scaler=None):
    """
    Complete preprocessing pipeline 
    """
    print("\n[Preprocessing] Running full preprocessing pipeline...")
    
    # Apply feature engineering
    df = engineer_lendingclub_features(df)
    
    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from scaling
    if 'risk' in numeric_cols:
        numeric_cols.remove('risk')
    
    # Scale features 
    if is_training:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        if scaler is not None:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df, scaler