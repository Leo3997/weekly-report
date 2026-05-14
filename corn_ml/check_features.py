import pandas as pd
import numpy as np
from build_features import build_features

df = build_features()
print('特征标准差（最近100天）:')
recent = df.tail(100)
for col in ['return_1d', 'volatility', 'rsi_14', 'ma5_ma20_ratio', 'month_cos']:
    print(f'  {col}: std={recent[col].std():.4f}, min={recent[col].min():.4f}, max={recent[col].max():.4f}')
