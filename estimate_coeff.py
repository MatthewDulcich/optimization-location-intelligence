"""
1. This script does the following:
- Creates baseline food access metric (baseline_L) using restaurant density
- Applies demographic weighting through normalized features and predefined correlations
- Balances baseline and weighted components to form composite metric (L)
- Enforces metric positivity and data stability with clipping/Nan handling
- Normalizes final metric to per-1,000 residents for cross-region comparison
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# 1. Calculate a baseline L using population and total restaurants (avoid 0)
# ----------------------------------------------------------------------------
# Avoid division by zero: add epsilon (1e-5) to totalPopulation and totalRestaurants
df['baseline_L'] = (
    (df['totalPopulation'] + 1e-5) * 
    (df['totalRestaurants'] + 1e-5) / 
    (df['totalPopulation'] + 1e-5)  # Simplified to totalRestaurants, but ensures > 0
)

# Identify restaurant percentage columns (ending with 'Percent')
restaurant_cols = [col for col in df.columns if col.endswith('Percent') and col != 'totalPopulation']

# Identify demographic columns
demographic_cols = [
    'hispanicOrLatinoPercent', 'blackPercent', 'asianNativeHawaiianOtherPacificIslanderPercent',
    'medianHouseholdIncome', 'aged65AndOlder', 'educationLessThanCollegePercent',
    'povertyPercent', 'urbanRuralStatus', 'medianHomeValueUSD', 'obesityPercent',
    'physicalInactivityPercent', 'totalPopulation', 'whitePercent'
]

# Initialize weighted_L with zeros
df['weighted_L'] = 0.0

for r_col, d_cols in weights_dict.items():
    for d_col, corr in d_cols:
        # Normalize demographic column to [0, 1], fill NaN with 0
        d_min = df[d_col].min()
        d_max = df[d_col].max()
        d_range = d_max - d_min + 1e-10  # Avoid division by zero
        d_norm = (df[d_col].fillna(0) - d_min) / d_range
        
        # Add weighted contribution (replace NaN with 0)
        contribution = df[r_col].fillna(0) * d_norm * np.abs(corr)
        df['weighted_L'] += contribution.fillna(0)

# Scale weighted_L to avoid dominance over baseline
df['weighted_L'] = df['weighted_L'] * 0.5  # Adjust scaling factor as needed

# 3. Combine baseline and weighted contributions
# ----------------------------------------------
df['L'] = df['baseline_L'] + df['weighted_L']

# 4. Ensure L is strictly positive (even if baseline_L is 0)
# -----------------------------------------------------------
# Set a floor value (e.g., 1) to avoid L = 0
df['L'] = df['L'].clip(lower=1.0)  # Minimum L = 1

# 5. Normalize L to a meaningful scale (e.g., per 1,000 people)
# -------------------------------------------------------------
df['L_normalized'] = df['L'] / (df['totalPopulation'] + 1e-5) * 1000  # Avoid division by zero

