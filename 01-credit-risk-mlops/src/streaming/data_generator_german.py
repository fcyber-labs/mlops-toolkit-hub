"""
German Credit Data Generator - Creates synthetic data matching Phase 1 format
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path


class GermanCreditDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        # German Credit specific distributions
        self.age_dist = (35, 12)
        self.credit_amount_dist = (5000, 3000)
        self.duration_dist = (20, 12)
        self.purpose_options = [
            "car",
            "domestic appliances",
            "education",
            "furniture/equipment",
            "radio/TV",
            "repairs",
            "vacation/others",
        ]
        self.purpose_probs = [0.2, 0.1, 0.1, 0.15, 0.1, 0.15, 0.2]
        self.housing_options = ["own", "rent", "free"]
        self.housing_probs = [0.4, 0.4, 0.2]
        self.job_options = [1, 2, 3, 4]
        self.job_probs = [0.2, 0.3, 0.3, 0.2]

    def generate_loan(self, day=0):
        """Generate a single loan in German Credit format"""

        # Calculate drift factor
        drift_factor = 1 + (day / 1000) * 0.1

        loan = {
            "Age": max(18, min(80, np.random.normal(self.age_dist[0], self.age_dist[1]))),
            "Credit amount": max(
                1000,
                min(
                    20000,
                    np.random.normal(
                        self.credit_amount_dist[0] * drift_factor,
                        self.credit_amount_dist[1],
                    ),
                ),
            ),
            "Duration": max(
                6,
                min(72, np.random.normal(self.duration_dist[0], self.duration_dist[1])),
            ),
            "Job": np.random.choice(self.job_options, p=self.job_probs),
            "Purpose": np.random.choice(self.purpose_options, p=self.purpose_probs),
            "Housing": np.random.choice(self.housing_options, p=self.housing_probs),
            "Sex": np.random.choice(["male", "female"], p=[0.7, 0.3]),
        }

        return loan

    def generate_batch(self, n_loans=100, start_day=0):
        """Generate a batch of loans"""
        loans = [self.generate_loan(start_day + i) for i in range(n_loans)]
        return pd.DataFrame(loans)

    def generate_stream(self, days=30, loans_per_day=50, output_dir="data/streaming/german"):
        """Generate streaming data over multiple days"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_loans = []

        for day in range(days):
            df_day = self.generate_batch(loans_per_day, day)
            df_day["day"] = day
            df_day["timestamp"] = datetime.now() - timedelta(days=days - day)
            all_loans.append(df_day)

            filename = f"{output_dir}/loans_day_{day:03d}.csv"
            df_day.to_csv(filename, index=False)
            print(f"Day {day:3d}: Generated {len(df_day)} loans")

        df_all = pd.concat(all_loans, ignore_index=True)
        df_all.to_csv(f"{output_dir}/all_loans.csv", index=False)

        print(f"\n✅ Generated {len(df_all)} German Credit loans over {days} days")
        print(f"Columns: {list(df_all.columns)}")
        return df_all


if __name__ == "__main__":
    generator = GermanCreditDataGenerator()
    df = generator.generate_stream(days=5, loans_per_day=10)
    print("\nSample data:")
    print(df.head())
