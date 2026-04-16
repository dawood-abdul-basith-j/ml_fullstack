"""
=============================================================
  DEMAND FORECASTING — ARTIFACT EXPORT SCRIPT
  Run this cell at the END of your Colab notebook
  (after the model has been trained and `history` exists)
=============================================================
Outputs:
  - forecasts.json          (actual vs predicted per entity)
  - training_history.json   (loss curves per epoch)
  - model_metadata.json     (KPIs, entity list, per-entity MAE)

Then copy all 3 JSON files into website/ folder beside index.html
=============================================================
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Rebuild preprocessing (same as notebook) ──────────────────────────────────
df = pd.read_csv('demand_forecasting_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

df_encoded = pd.get_dummies(df, columns=['store', 'product'], prefix=['store', 'prod'])
df_encoded['store'] = df['store']
df_encoded['product'] = df['product']

dates_sorted = df_encoded['date'].sort_values().unique()
train_cutoff = dates_sorted[int(len(dates_sorted) * 0.70)]
val_cutoff   = dates_sorted[int(len(dates_sorted) * 0.85)]

train_df = df_encoded[df_encoded['date'] <= train_cutoff]
test_df  = df_encoded[df_encoded['date'] > val_cutoff]

exclude_cols = ['date', 'store', 'product', 'demand']
features     = ['demand'] + [c for c in df_encoded.columns if c not in exclude_cols]
target_index = 0
lookback_days = 30

scaler = MinMaxScaler()
scaler.fit(train_df[features])

demand_min   = float(scaler.data_min_[target_index])
demand_max   = float(scaler.data_max_[target_index])
demand_range = demand_max - demand_min

# ── Run predictions per entity ────────────────────────────────────────────────
forecasts      = {}
per_entity_mae = {}
all_true, all_pred = [], []

for (store, product), group in test_df.groupby(['store', 'product']):
    group = group.sort_values('date')
    if len(group) <= lookback_days:
        continue

    scaled = scaler.transform(group[features])
    X, y_true_s, date_list = [], [], []

    for i in range(len(scaled) - lookback_days):
        X.append(scaled[i : i + lookback_days])
        y_true_s.append(scaled[i + lookback_days, target_index])
        date_list.append(str(group['date'].iloc[i + lookback_days].date()))

    X = np.array(X)
    y_pred_s = model.predict(X, verbose=0).flatten()

    y_true_actual = (np.array(y_true_s) * demand_range + demand_min).round(2)
    y_pred_actual = (y_pred_s           * demand_range + demand_min).round(2)

    key = f"{store} - {product}"
    forecasts[key] = {
        "store":     store,
        "product":   product,
        "dates":     date_list,
        "actual":    y_true_actual.tolist(),
        "predicted": y_pred_actual.tolist(),
    }
    mae_val = round(mean_absolute_error(y_true_actual, y_pred_actual), 2)
    per_entity_mae[key] = mae_val
    all_true.extend(y_true_actual.tolist())
    all_pred.extend(y_pred_actual.tolist())

# ── Overall metrics ───────────────────────────────────────────────────────────
overall_mae  = round(mean_absolute_error(all_true, all_pred), 4)
overall_rmse = round(float(np.sqrt(mean_squared_error(all_true, all_pred))), 4)
overall_r2   = round(r2_score(all_true, all_pred), 4)

# ── Dataset cost constants (from CSV) ────────────────────────────────────────
holding_cost_per_unit  = float(df['holding_cost_per_unit'].mean())
stockout_cost_per_unit = float(df['stockout_cost_per_unit'].mean())
avg_demand             = float(df['demand'].mean())

metadata = {
    "overall_mae":           overall_mae,
    "overall_rmse":          overall_rmse,
    "overall_r2":            overall_r2,
    "total_samples":         len(all_true),
    "num_entities":          len(forecasts),
    "entities":              list(forecasts.keys()),
    "per_entity_mae":        per_entity_mae,
    "lookback_days":         lookback_days,
    "holding_cost_per_unit": holding_cost_per_unit,
    "stockout_cost_per_unit":stockout_cost_per_unit,
    "avg_demand":            round(avg_demand, 2),
    "demand_min":            round(demand_min, 2),
    "demand_max":            round(demand_max, 2),
}

# ── Training history ──────────────────────────────────────────────────────────
history_data = {
    "loss":     [round(v, 6) for v in history.history['loss']],
    "val_loss": [round(v, 6) for v in history.history['val_loss']],
    "mae":      [round(v, 6) for v in history.history.get('mae', [])],
    "val_mae":  [round(v, 6) for v in history.history.get('val_mae', [])],
}

# ── Save JSONs ────────────────────────────────────────────────────────────────
with open('model_metadata.json',   'w') as f: json.dump(metadata,      f, indent=2)
with open('forecasts.json',        'w') as f: json.dump(forecasts,     f, indent=2)
with open('training_history.json', 'w') as f: json.dump(history_data,  f, indent=2)

print("✅  Artifacts exported successfully!")
print(f"   Overall MAE  : {overall_mae}")
print(f"   Overall RMSE : {overall_rmse}")
print(f"   Overall R²   : {overall_r2}")
print(f"   Entities     : {len(forecasts)}")
print()
print("📁  Copy these files into  website/  folder:")
print("     model_metadata.json")
print("     forecasts.json")
print("     training_history.json")
