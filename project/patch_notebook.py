"""
patch_notebook.py
─────────────────────────────────────────────────────────────
Run this script ONCE from the same folder as your notebook:

    python patch_notebook.py

It appends a new "Export Artifacts" cell to:
    production_ready_dl_v5.ipynb

The patched notebook is saved as:
    production_ready_dl_v5_with_export.ipynb

Upload that file to Colab, run all cells, then download
the 3 JSON files and drop them in  website/
─────────────────────────────────────────────────────────────
"""

import json, pathlib, sys

NOTEBOOK_PATH = pathlib.Path("production_ready_dl_v5.ipynb")
OUTPUT_PATH   = pathlib.Path("production_ready_dl_v5_with_export.ipynb")

# ── The export cell source ────────────────────────────────────────────────────
EXPORT_CELL_SOURCE = r"""# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. EXPORT ARTIFACTS FOR WEB DASHBOARD                          ║
# ║  Saves model_metadata.json, forecasts.json, training_history.json ║
# ║  Copy those 3 files into the  website/  folder.                 ║
# ╚══════════════════════════════════════════════════════════════════╝

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Rebuild preprocessing (identical to training pipeline) ──────────────────
_df = pd.read_csv('demand_forecasting_dataset.csv')
_df['date'] = pd.to_datetime(_df['date'])

_df_enc = pd.get_dummies(_df, columns=['store', 'product'], prefix=['store', 'prod'])
_df_enc['store']   = _df['store']
_df_enc['product'] = _df['product']

_dates_sorted = _df_enc['date'].sort_values().unique()
_train_cutoff = _dates_sorted[int(len(_dates_sorted) * 0.70)]
_val_cutoff   = _dates_sorted[int(len(_dates_sorted) * 0.85)]

_train_df = _df_enc[_df_enc['date'] <= _train_cutoff]
_test_df  = _df_enc[_df_enc['date'] >  _val_cutoff]

_excl    = ['date', 'store', 'product', 'demand']
_features = ['demand'] + [c for c in _df_enc.columns if c not in _excl]
_tidx     = 0          # target column index (demand)
_lookback = 30         # same as training

_sc = MinMaxScaler()
_sc.fit(_train_df[_features])

_dmin   = float(_sc.data_min_[_tidx])
_dmax   = float(_sc.data_max_[_tidx])
_drange = _dmax - _dmin

# ── Run predictions per store-product entity ─────────────────────────────────
_forecasts      = {}
_per_entity_mae = {}
_all_true, _all_pred = [], []

for (_store, _product), _grp in _test_df.groupby(['store', 'product']):
    _grp = _grp.sort_values('date')
    if len(_grp) <= _lookback:
        continue

    _scaled = _sc.transform(_grp[_features])
    _X, _yt, _dates = [], [], []

    for i in range(len(_scaled) - _lookback):
        _X.append(_scaled[i : i + _lookback])
        _yt.append(_scaled[i + _lookback, _tidx])
        _dates.append(str(_grp['date'].iloc[i + _lookback].date()))

    _X  = np.array(_X)
    _yp = model.predict(_X, verbose=0).flatten()

    _yt_real = (np.array(_yt) * _drange + _dmin).round(2)
    _yp_real = (_yp           * _drange + _dmin).round(2)

    _key = f"{_store} - {_product}"
    _forecasts[_key] = {
        "store":     _store,
        "product":   _product,
        "dates":     _dates,
        "actual":    _yt_real.tolist(),
        "predicted": _yp_real.tolist(),
    }
    _mae = round(mean_absolute_error(_yt_real, _yp_real), 2)
    _per_entity_mae[_key] = _mae
    _all_true.extend(_yt_real.tolist())
    _all_pred.extend(_yp_real.tolist())

# ── Overall metrics ──────────────────────────────────────────────────────────
_o_mae  = round(mean_absolute_error(_all_true, _all_pred), 4)
_o_rmse = round(float(np.sqrt(mean_squared_error(_all_true, _all_pred))), 4)
_o_r2   = round(r2_score(_all_true, _all_pred), 4)

_metadata = {
    "overall_mae":           _o_mae,
    "overall_rmse":          _o_rmse,
    "overall_r2":            _o_r2,
    "total_samples":         len(_all_true),
    "num_entities":          len(_forecasts),
    "entities":              list(_forecasts.keys()),
    "per_entity_mae":        _per_entity_mae,
    "lookback_days":         _lookback,
    "holding_cost_per_unit": float(_df['holding_cost_per_unit'].mean()),
    "stockout_cost_per_unit":float(_df['stockout_cost_per_unit'].mean()),
    "avg_demand":            round(float(_df['demand'].mean()), 2),
    "demand_min":            round(_dmin, 2),
    "demand_max":            round(_dmax, 2),
}

_hist_data = {
    "loss":    [round(v, 6) for v in history.history['loss']],
    "val_loss":[round(v, 6) for v in history.history['val_loss']],
    "mae":     [round(v, 6) for v in history.history.get('mae', [])],
    "val_mae": [round(v, 6) for v in history.history.get('val_mae', [])],
}

# ── Write JSON files ─────────────────────────────────────────────────────────
with open('model_metadata.json',   'w') as _f: json.dump(_metadata,  _f, indent=2)
with open('forecasts.json',        'w') as _f: json.dump(_forecasts, _f, indent=2)
with open('training_history.json', 'w') as _f: json.dump(_hist_data, _f, indent=2)

print("✅  Artifacts exported!")
print(f"   Overall MAE  : {_o_mae}")
print(f"   Overall RMSE : {_o_rmse}")
print(f"   Overall R²   : {_o_r2}")
print(f"   Entities     : {len(_forecasts)}")
print()
print("📥  Download these 3 files from the Colab file browser (left panel):")
print("     model_metadata.json")
print("     forecasts.json")
print("     training_history.json")
print()
print("📂  Then paste them into  project/website/  alongside index.html")
"""

# ── Patch the notebook ────────────────────────────────────────────────────────
if not NOTEBOOK_PATH.exists():
    print(f"ERROR: '{NOTEBOOK_PATH}' not found in current directory.")
    print(f"       Run this script from inside the project/ folder.")
    sys.exit(1)

nb = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))

# Build the new cell in nbformat v4 style
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in EXPORT_CELL_SOURCE.splitlines()],
}

# Also add a markdown header cell before it
header_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 6. Export Artifacts for Web Dashboard\n",
               "Run this cell after training to generate the JSON files needed by the website."],
}

nb["cells"].append(header_cell)
nb["cells"].append(new_cell)

OUTPUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

print("[ OK ] Notebook patched successfully!")
print(f"   Input  : {NOTEBOOK_PATH}")
print(f"   Output : {OUTPUT_PATH}")
print()
print("Next steps:")
print("  1. Upload  production_ready_dl_v5_with_export.ipynb  to Google Colab")
print("  2. Runtime -> Run all  (or Ctrl+F9)")
print("  3. Download the 3 JSON files from Colab's file browser")
print("  4. Copy them into  project/website/")
