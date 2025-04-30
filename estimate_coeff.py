import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Load your DataFrame
data = pd.read_csv("data/restaurant_data.csv")

# ======================================================================
# Step 1: Preprocess Data (Handle Missing Values and Zero Variance)
# ======================================================================
# Define demographic columns first to avoid overlap with restaurant_cols
demographic_cols = [
    'hispanicOrLatinoPercent', 'blackPercent', 'asianNativeHawaiianOtherPacificIslanderPercent',
    'medianHouseholdIncome', 'aged65AndOlder', 'educationLessThanCollegePercent',
    'povertyPercent', 'urbanRuralStatus', 'medianHomeValueUSD', 'obesityPercent',
    'physicalInactivityPercent', 'totalPopulation', 'whitePercent'
]

# Define restaurant columns as Percent columns NOT in demographic_cols
restaurant_cols = [col for col in data.columns if col.endswith('Percent') and col not in demographic_cols]

# Impute missing values with 0 for simplicity (adjust if better methods exist)
data[restaurant_cols] = data[restaurant_cols].fillna(0)
data[demographic_cols] = data[demographic_cols].fillna(0)

# ======================================================================
# Step 2: Compute Baseline L (Guaranteed Non-Zero)
# ======================================================================
# Avoid division by zero: add epsilon to totalPopulation and totalRestaurants
epsilon = 1e-5
df = data.copy()
df['baseline_L'] = (
    (df['totalPopulation'] + epsilon) * 
    (df['totalRestaurants'] + epsilon)
)
df['baseline_L'] = df['baseline_L'].replace(0, epsilon)  # Ensure no zeros

# ======================================================================
# Step 3: Compute Significant Correlations (Skip Invalid/Zero-Variance Pairs)
# ======================================================================
results = []
for r_col in restaurant_cols:
    for d_col in demographic_cols:
        subset = df[[r_col, d_col]].dropna()
        if len(subset) < 10:
            continue
        # Calculate variances as scalars
        var_r = subset[r_col].var()
        var_d = subset[d_col].var()
        # Skip if any variance is too low (use absolute value checks)
        if var_r < 1e-10 or var_d < 1e-10:
            continue
        try:
            corr, p_value = pearsonr(subset[r_col], subset[d_col])
            if np.isnan(corr) or np.isinf(corr):
                continue
            results.append((r_col, d_col, corr, p_value))
        except:
            continue

corr_df = pd.DataFrame(results, columns=['RestaurantType', 'Demographic', 'Correlation', 'PValue'])
corr_df = corr_df.dropna()

# Bonferroni adjustment
alpha = 0.05
n_tests = len(corr_df)
corr_df['PValueAdj'] = np.minimum(corr_df['PValue'] * n_tests, 1.0)
corr_df['Significant'] = corr_df['PValueAdj'] < alpha

# Filter strong correlations
strong_corr_df = corr_df[
    (corr_df['Significant']) & 
    (np.abs(corr_df['Correlation']) > 0.3)
].sort_values(by='Correlation', ascending=False)

# ======================================================================
# Step 4: Calculate Weighted Contributions (Handle Zero Variance in Normalization)
# ======================================================================
weights_dict = {}
for _, row in strong_corr_df.iterrows():
    r_col = row['RestaurantType']
    d_col = row['Demographic']
    corr = row['Correlation']
    if r_col not in weights_dict:
        weights_dict[r_col] = []
    weights_dict[r_col].append((d_col, corr))

df['weighted_L'] = 0.0

for r_col, d_cols in weights_dict.items():
    for d_col, corr in d_cols:
        # Normalize demographic column safely
        d_min = df[d_col].min()
        d_max = df[d_col].max()
        d_range = d_max - d_min
        if d_range < 1e-10:  # Skip zero-variance demographics
            continue
        d_norm = (df[d_col] - d_min) / d_range
        contribution = df[r_col] * d_norm * np.abs(corr)
        df['weighted_L'] += contribution.fillna(0)

# Scale weighted_L to avoid dominance over baseline
df['weighted_L'] = df['weighted_L'] * 0.5  # Tune this factor

# ======================================================================
# Step 5: Combine Baseline and Weighted_L (Ensure L > 0 and No NaNs)
# ======================================================================
df['L'] = df['baseline_L'] + df['weighted_L']
df['L'] = df['L'].fillna(df['baseline_L'])  # Fallback to baseline if NaN
df['L'] = df['L'].replace(0, epsilon)  # Replace 0 with epsilon

# View results
print(df[['fips', 'totalPopulation', 'totalRestaurants', 'baseline_L', 'weighted_L', 'L']].head())
# Save the DataFrame with L values
df.to_csv("data/data_with_L.csv", index=False)