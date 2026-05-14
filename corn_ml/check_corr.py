import pandas as pd
import numpy as np

df = pd.read_csv('features_full_v2.csv', parse_dates=['date'])
recent = df.tail(500)

from build_features import FEATURE_COLS

corr_matrix = recent[FEATURE_COLS].corr()

high_corr_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        corr = abs(corr_matrix.iloc[i, j])
        if corr > 0.7:
            high_corr_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], corr))

high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

print("高相关特征对 (|corr| > 0.7):")
for f1, f2, c in high_corr_pairs[:30]:
    print(f"  {f1} <-> {f2}: {c:.3f}")
