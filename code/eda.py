import pandas as pd
from skimpy import skim
import numpy as np
import sys

pd.set_option('display.float_format', '{:.4f}'.format)

df = pd.read_excel("data/DryBeanDataSet.xlsx")

df['Constantness'] = df['Constantness'].astype(str)
df['Extent'] = df['Extent'].replace("?", np.nan)
df['Compactness'] = df['Compactness'].replace("?", np.nan)
df['ShapeFactor6'] = df['ShapeFactor6'].replace("?", np.nan)
#df.to_csv("data/data.csv", index=False)

#cols = df.columns
#print(cols)

# name
# data type
# missing value percentage
# missing value description
# outliers (quartiles)
# skewness
# kurtosis
# SNR

# KNN
# Distance measures sensitive to outliers (small k should be fine).
# Missing values, impute if possible, otherwise ignore in distance calculation.
# Imbalanced datasets -> majority class will dominate.
#   - shepards method to fix.
# Normalise.
# sensitive to noise (output).


# Classification trees
# robust to outliers
# robust to noise.
# Sensitive to skew class distributions.
# Robust to missing values.

f = df[sys.argv[1]]

print(f"Stats for {sys.argv[1]}")

print("Distribution")
value_counts = f.value_counts()
print(value_counts)

if f.dtype != object:
    print(f"Sum missing 0 {sum(f == 0)}")
    print(f"Sum missing -1 {sum(f == -1)}")
    print(f"Sum missing < 0 {sum(f < 0)}")
    print(f"Sum nan {sum(f.isnull())}")
    missing_perc = sum(f.isnull()) / value_counts.sum() * 100
    print(f"? % {missing_perc}")

    print("Quartiles")
    print(f.describe())
    print(f"Skewness {f.skew()}")
    print(f"Kurtosis {f.kurt()}")

    def calculate_snr_db(data):
        signal = np.mean(data)
        noise = data - signal
        snr = np.mean(signal**2) / np.mean(noise**2)
        return 10 * np.log10(snr)

    print(f"SNR {calculate_snr_db(f)}")

else:
    def gini_impurity(series):
        p = series.value_counts() / len(series)
        return 1 - np.sum(p**2)
    print(f"Gini impurity {gini_impurity(f)}")

    imbalance_ratio = value_counts.max() / value_counts.min()
    print(f"Imbalance ratio {imbalance_ratio}")

    majority_perc = (value_counts.max() / value_counts.sum()) * 100
    print(f"Majority % {majority_perc}")

    missing_perc = sum(f == '?') / value_counts.sum() * 100

    print(f"? % {missing_perc}")

# SNR 0 - 10 -> poor, 10 - 20 -> acceptable
# KURT 0 -> normal dist, > 0, heavy tail, < 0 -> light tailed.
# |SKEW| < 0.5 -> ~ symmetric, 0.5 to 1 -> moderately skewed, > 1 highly skewed.

stats = pd.read_csv("data/stats.csv")
    














