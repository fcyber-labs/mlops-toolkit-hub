"""
Synthetic Data Generator - Creates realistic loan data with proper numeric types
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

class LoanDataGenerator:
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Base distributions
        self.age_dist = (35, 12)
        self.income_dist = (70000, 20000)
        self.credit_amount_dist = (15000, 8000)
        self.duration_probs = {36: 0.7, 60: 0.3}
        self.purpose_options = ['debt_consolidation', 'credit_card', 'home_improvement', 
                                'other', 'major_purchase', 'medical', 'car']
        self.purpose_probs = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
        self.housing_options = ['rent', 'mortgage', 'own']
        self.housing_probs = [0.4, 0.4, 0.2]
        self.housing_map = {'rent': 0, 'mortgage': 1, 'own': 2}
        
    def generate_loan(self, day=0):
        """Generate a single loan application with numeric values"""
        
        # Calculate drift factor
        drift_factor = 1 + (day / 1000) * 0.1
        
        housing_str = np.random.choice(self.housing_options, p=self.housing_probs)
        
        loan = {
            'age': float(max(18, min(80, np.random.normal(self.age_dist[0], self.age_dist[1])))),
            'credit_amount': float(max(1000, min(50000, np.random.normal(
                self.credit_amount_dist[0] * drift_factor, 
                self.credit_amount_dist[1])))),
            'duration': float(np.random.choice(list(self.duration_probs.keys()), 
                                        p=list(self.duration_probs.values()))),
            'purpose': np.random.choice(self.purpose_options, p=self.purpose_probs),
            'income': float(max(20000, min(300000, np.random.normal(
                self.income_dist[0], 
                self.income_dist[1])))),
            'emp_length': float(min(40, max(0, np.random.exponential(8)))),
            'housing': float(self.housing_map[housing_str]),
            'dti': float(min(50, max(0, np.random.gamma(2, 5)))),
            'int_rate': float(np.random.uniform(5, 25)),
        }
        
        return loan
    
    def generate_batch(self, n_loans=100, start_day=0):
        """Generate a batch of loans"""
        loans = [self.generate_loan(start_day + i) for i in range(n_loans)]
        return pd.DataFrame(loans)
    
    def generate_stream(self, days=30, loans_per_day=50, output_dir="data/streaming"):
        """Generate streaming data over multiple days"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_loans = []
        
        for day in range(days):
            df_day = self.generate_batch(loans_per_day, day)
            df_day['day'] = day
            df_day['timestamp'] = datetime.now() - timedelta(days=days-day)
            all_loans.append(df_day)
            
            filename = f"{output_dir}/loans_day_{day:03d}.csv"
            df_day.to_csv(filename, index=False)
            print(f"Day {day:3d}: Generated {len(df_day)} loans")
        
        df_all = pd.concat(all_loans, ignore_index=True)
        df_all.to_csv(f"{output_dir}/all_loans.csv", index=False)
        
        print(f"\n✅ Generated {len(df_all)} loans over {days} days")
        print(f"Columns: {list(df_all.columns)}")
        return df_all

if __name__ == "__main__":
    generator = LoanDataGenerator()
    df = generator.generate_stream(days=5, loans_per_day=10)
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)