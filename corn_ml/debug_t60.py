import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

# 加载数据
data = np.load('features_h60.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

# 训练模型
model = LGBMClassifier(
    n_estimators=400, learning_rate=0.03, max_depth=4,
    num_leaves=16, min_child_samples=50, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.01, reg_lambda=0.1,
    min_gain_to_split=0.0, is_unbalance=True,
    random_state=42, verbose=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc")

# 原始预测
raw_val = model.predict_proba(X_val)[:, 1]
raw_test = model.predict_proba(X_test)[:, 1]

print("原始概率分布:")
print(f"  val:  std={raw_val.std():.4f}, min={raw_val.min():.4f}, max={raw_val.max():.4f}")
print(f"  test: std={raw_test.std():.4f}, min={raw_test.min():.4f}, max={raw_test.max():.4f}")
print(f"  test唯一值数量: {np.unique(raw_test).shape[0]}")

# 尝试Platt校准
try:
    cal_platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    cal_platt.fit(X_val, y_val)
    platt_test = cal_platt.predict_proba(X_test)[:, 1]
    print(f"\nPlatt校准后:")
    print(f"  test: std={platt_test.std():.4f}, min={platt_test.min():.4f}, max={platt_test.max():.4f}")
    print(f"  test唯一值数量: {np.unique(platt_test).shape[0]}")
except Exception as e:
    print(f"\nPlatt校准失败: {e}")
