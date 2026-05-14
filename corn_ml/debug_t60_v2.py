import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss
import warnings

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

raw_val_c = np.clip(raw_val, 1e-6, 1 - 1e-6)
raw_test_c = np.clip(raw_test, 1e-6, 1 - 1e-6)

raw_spread = raw_test.std()
raw_brier = brier_score_loss(y_val, raw_val_c)

print(f"原始: test_std={raw_spread:.4f}, val_brier={raw_brier:.4f}")

best_cal = raw_test.copy()
best_brier = float('inf')
best_method = "none"

# 方法1: Platt scaling
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal_platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        cal_platt.fit(X_val, y_val)
        platt_cal = cal_platt.predict_proba(X_test)[:, 1]
        platt_brier = brier_score_loss(y_val, cal_platt.predict_proba(X_val)[:, 1])
        print(f"Platt: test_std={platt_cal.std():.4f}, val_brier={platt_brier:.4f}")
        if platt_brier < best_brier:
            best_brier = platt_brier
            best_cal = platt_cal
            best_method = "Platt scaling"
except Exception as e:
    print(f"Platt失败: {e}")

# 方法2: 温度缩放
def _brier_t(T):
    p = expit(logit(raw_val_c) / T)
    return brier_score_loss(y_val, p)

try:
    res = minimize_scalar(_brier_t, bounds=(0.2, 5.0), method="bounded")
    T_opt = float(res.x)
    temp_cal = expit(logit(raw_test_c) / T_opt)
    temp_cal = np.clip(temp_cal, 0.001, 0.999)
    temp_val_cal = expit(logit(raw_val_c) / T_opt)
    temp_val_cal = np.clip(temp_val_cal, 0.001, 0.999)
    temp_brier = brier_score_loss(y_val, temp_val_cal)
    print(f"温度: T={T_opt:.2f}, test_std={temp_cal.std():.4f}, val_brier={temp_brier:.4f}")
    if temp_brier < best_brier:
        best_brier = temp_brier
        best_cal = temp_cal
        direction = "锐化" if T_opt < 1.0 else "平滑"
        best_method = f"温度缩放(T={T_opt:.2f},{direction})"
except Exception as e:
    print(f"温度失败: {e}")

cal_spread = best_cal.std()
improvement = (raw_brier - best_brier) / raw_brier * 100 if raw_brier > 0 else 0

print(f"\n最佳方法: {best_method}")
print(f"改善: {improvement:.1f}%")
print(f"标准差: {raw_spread:.4f} -> {cal_spread:.4f}")
print(f"跳过检查: cal_spread < raw_spread * 0.3 = {cal_spread:.4f} < {raw_spread * 0.3:.4f} = {cal_spread < raw_spread * 0.3}")

if cal_spread < raw_spread * 0.3:
    print(">>> 跳过校准 (过度平滑)")
elif improvement < 5.0 and best_method != "none":
    print(f">>> 跳过校准 (改善不足)")
else:
    print(f">>> 使用校准: {best_method}")
