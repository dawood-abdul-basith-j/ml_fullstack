# Demand Forecasting for Inventory Cost Reduction
## Comprehensive Project Report — Sem 4 Machine Learning

---

## 1. Problem Statement

Retail businesses carry two opposing costs that squeeze profitability:

| Cost | Cause | Impact |
|---|---|---|
| **Holding Cost** | Overstocking — ordering too much | ₹1.64/unit/day (dataset average) |
| **Stockout Cost** | Understocking — running out of stock | ₹12.30/unit (missed sale penalty) |

**Goal:** Predict next-day demand per store-product combination with high accuracy so that inventory managers can order *just enough* — eliminating waste without causing stockouts.

**Why Deep Learning?** Traditional methods (ARIMA, simple regression) struggle with:
- Multiple simultaneous time series (15 store-product entities)
- Complex non-linear relationships between features
- Long-range temporal dependencies in daily demand patterns

---

## 2. Dataset Overview

**File:** `demand_forecasting_dataset.csv` (660 KB)

| Attribute | Detail |
|---|---|
| **Stores** | 3 (Store_A, Store_B, Store_C) |
| **Products** | 5 (Clothing, Electronics, Furniture, Groceries, Toys) |
| **Entities** | 15 unique store-product combinations |
| **Time period** | Daily records (continuous) |
| **Target variable** | `demand` — units sold per day |

### Raw Feature Columns

| Feature | Type | Description |
|---|---|---|
| `date` | Datetime | Calendar date |
| `store` | Categorical | Store identifier |
| `product` | Categorical | Product category |
| `demand` | Numeric | Units sold (TARGET) |
| `is_promotion` | Binary | Promotional event active (0/1) |
| `unit_cost` | Numeric | Cost per unit purchased |
| `selling_price` | Numeric | Retail selling price |
| `holding_cost_per_unit` | Numeric | Storage cost per unit per day |
| `stockout_cost_per_unit` | Numeric | Penalty cost when demand exceeds stock |
| `temperature` | Numeric | Daily temperature (environmental factor) |
| `is_weekend` | Binary | Weekend flag (0/1) |
| `is_holiday` | Binary | Public holiday flag (0/1) |

---

## 3. ML Pipeline — Step by Step

### Step 3.1 — Feature Engineering: One-Hot Encoding (OHE)

**Problem:** Neural networks require numeric input. `store` and `product` are categorical strings.

**Solution — One-Hot Encoding:**
```python
df_encoded = pd.get_dummies(df, columns=['store', 'product'],
                            prefix=['store', 'prod'])
```

This converts the categorical columns into binary indicator columns:

| store_Store_A | store_Store_B | store_Store_C | prod_Clothing | prod_Electronics | ... |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 1 | ... |
| 0 | 1 | 0 | 1 | 0 | ... |

**Result:** 3 stores + 5 products + 9 numerical features = **17 total features** per timestep.

> [!NOTE]
> `store` and `product` string columns are preserved separately for grouping purposes even after encoding. They are explicitly excluded from the feature vector fed to the model.

---

### Step 3.2 — Data Splitting: Strict Chronological Split

**Critical design decision:** Time series data must NEVER be split randomly.

**Why random splits are wrong for time series:**
If you train on data from Jan–Dec and test on Feb (which is in the middle), the model has already "seen the future" relative to Feb. This causes data leakage — artificially inflated accuracy that won't hold in production.

**Our chronological split (by unique dates):**
```
|←———— 70% Train ————→|←— 15% Val —→|←— 15% Test —→|
Day 1, 2, 3 ...        ...            ...              Last Day
```

```python
dates         = df_encoded['date'].sort_values().unique()
train_cutoff  = dates[int(len(dates) * 0.70)]
val_cutoff    = dates[int(len(dates) * 0.85)]

train_df = df_encoded[df_encoded['date'] <= train_cutoff]
val_df   = df_encoded[(df_encoded['date'] > train_cutoff)
                    & (df_encoded['date'] <= val_cutoff)]
test_df  = df_encoded[df_encoded['date'] > val_cutoff]
```

| Split | Purpose | Sequences |
|---|---|---|
| **Train (70%)** | Model learns weights | 7,215 |
| **Validation (15%)** | Tune hyperparameters, prevent overfitting | 1,200 |
| **Test (15%)** | Final unbiased evaluation | 1,185 |

---

### Step 3.3 — Normalisation: MinMaxScaler

**Problem:** Features have vastly different ranges. `demand` might be 0–408 while `is_weekend` is only 0–1. Neural networks are sensitive to feature scale — large-valued features dominate gradient updates.

**Solution — Min-Max Normalisation:**

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

All values are compressed to the [0, 1] range.

```python
scaler = MinMaxScaler()
scaler.fit(train_df[features])   # fit ONLY on training data
```

> [!IMPORTANT]
> The scaler is fitted **only on training data**. This is critical. If you fit on the full dataset (including validation/test), future price/demand statistics "leak" into your scaler — a subtle form of data leakage. The same fitted scaler is then applied (`.transform()`) to validation and test data.

**Feature order in the vector (17 dimensions):**
```
[demand, is_promotion, unit_cost, selling_price, holding_cost_per_unit,
 stockout_cost_per_unit, temperature, is_weekend, is_holiday,
 store_Store_A, store_Store_B, store_Store_C,
 prod_Clothing, prod_Electronics, prod_Furniture, prod_Groceries, prod_Toys]
```

`demand` is at index 0 — both an input feature (past values) and the prediction target (next-day value).

---

### Step 3.4 — Entity-Aware Sequence Generation (Sliding Window)

**The Core Concept:** LSTMs require sequences of historical data as input, not individual rows.

**Lookback window = 30 days:**
The model sees the last 30 days of all 17 features to predict day 31's demand.

```
Input Window (30 timesteps × 17 features)          Output
[Day 1: demand, price, store_OHE, ...]
[Day 2: demand, price, store_OHE, ...]     →→→    Demand on Day 31
...
[Day 30: demand, price, store_OHE, ...]
```

**Entity-Aware — why this matters:**
We CANNOT mix sequences across different store-product pairs. Store_A-Groceries has completely different demand patterns than Store_C-Electronics. Mixing would corrupt the temporal sequence.

```python
def process_groups_into_sequences(df_subset, scaler, features, lookback, ...):
    for (store, product), group in df_subset.groupby(['store', 'product']):
        group = group.sort_values('date')        # chronological order
        scaled_data = scaler.transform(group[features])

        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i : i + lookback])          # 30-day window
            y.append(scaled_data[i + lookback, target_index])  # next demand
```

**Final tensor shapes:**
```
X_train : (7215, 30, 17)   — 7215 sequences, each 30 days × 17 features
y_train : (7215,)           — 7215 next-day demand values (scaled)
```

---

## 4. Model Architecture: Bidirectional LSTM

### 4.1 What is an LSTM?

A **Long Short-Term Memory** network is a special type of Recurrent Neural Network (RNN) designed to learn long-range dependencies in sequential data.

**The core problem with vanilla RNNs:** The vanishing gradient problem. When backpropagating errors through many timesteps, gradients shrink exponentially, making it impossible to learn from events far in the past.

**LSTM solution — the gating mechanism:**  
Each LSTM cell has three gates and a cell state:

| Gate | Formula | Role |
|---|---|---|
| **Forget Gate** (fₜ) | σ(Wf·[hₜ₋₁, xₜ] + bf) | Decides what old information to throw away |
| **Input Gate** (iₜ) | σ(Wi·[hₜ₋₁, xₜ] + bi) | Decides what new information to store |
| **Output Gate** (oₜ) | σ(Wo·[hₜ₋₁, xₜ] + bo) | Decides what to output from the cell |
| **Cell State** (Cₜ) | fₜ⊙Cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ]) | Long-term memory highway |

Where σ = sigmoid function, ⊙ = element-wise multiplication.

The **cell state** acts as a "conveyor belt" of memory — information can flow through many timesteps unchanged if the forget gate stays open.

---

### 4.2 What Makes it *Bidirectional*?

A standard LSTM processes time only **forward** (Day 1 → Day 2 → ... → Day 30).

A **Bidirectional LSTM** (Bi-LSTM) runs **two LSTMs in parallel**:
1. Forward: Day 1 → Day 30
2. Backward: Day 30 → Day 1

Their outputs are concatenated at each timestep:
```
Output = [forward_hidden ; backward_hidden]
```

**Why does this help for demand forecasting?**
The backward pass allows each timestep to "look ahead" in its sequence window, capturing both preceding and following contextual patterns within the 30-day lookback window. For example, a sales dip on day 20 has more context knowing that days 21-30 show a returning trend.

---

### 4.3 Our Model Architecture

```python
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(30, 17)),    # Layer 1
    Dropout(0.2),                           # Regularisation 1
    Bidirectional(LSTM(32)),                # Layer 2
    Dropout(0.2),                           # Regularisation 2
    Dense(16, activation='relu'),           # Dense 1
    Dense(1)                                # Output
])
```

**Layer-by-layer breakdown:**

| Layer | Output Shape | Parameters | Purpose |
|---|---|---|---|
| Bi-LSTM(64) | (30, 128) | ~33K | Extract temporal features, 64 units × 2 directions = 128 |
| Dropout(0.2) | (30, 128) | 0 | Drop 20% of neurons randomly during training to prevent overfitting |
| Bi-LSTM(32) | (64,) | ~24K | Compress to final representation, 32 × 2 = 64 |
| Dropout(0.2) | (64,) | 0 | Regularisation |
| Dense(16, ReLU) | (16,) | ~1K | Non-linear transformation |
| Dense(1) | (1,) | 17 | Single demand value output |

**Total trainable parameters:** ~58K (relatively lightweight but powerful)

**`return_sequences=True`** in Layer 1 means it outputs the hidden state at **every timestep** (shape 30×128), not just the last one. This feeds the full temporal sequence into Layer 2.

---

### 4.4 Loss Function & Optimiser

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

**Loss = Mean Squared Error (MSE):**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{true})^2$$

Squaring penalises large errors more severely, pushing the model to avoid big misses.

**Optimiser = Adam (Adaptive Moment Estimation):**
Adam combines:
- **Momentum:** Uses exponential moving average of past gradients
- **RMSprop:** Adapts learning rate per-parameter

It's the de-facto standard for deep learning — fast convergence, robust to noisy gradients.

---

## 5. Training Strategy: Smart Callbacks

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

checkpoint = ModelCheckpoint(
    'best_bi_lstm_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```

### EarlyStopping
Monitors validation loss. If it doesn't improve for 10 consecutive epochs, training stops and weights are **restored to the best epoch** (not the last epoch, which may have started overfitting).

This prevents the model from memorising the training set.

### ReduceLROnPlateau (Learning Rate Scheduling)
If validation loss plateaus for 5 epochs, the learning rate is **halved** (factor=0.5).

From the actual training logs, LR reduction events:
- Epoch 17: LR dropped 0.001 → 0.0005
- Epoch 25: LR dropped 0.0005 → 0.00025
- Epoch 37: LR dropped → 0.000125
- Epoch 42: LR dropped → 0.0000625

This "annealing" allows finer weight adjustments as training converges.

### ModelCheckpoint
Saves the model state whenever validation loss hits a new minimum. Even if training continues (before EarlyStopping triggers), the best-ever model is preserved.

**Training ran all 50 epochs** — EarlyStopping did NOT trigger (val_loss kept slowly improving from epoch 33 onwards, just not fast enough for EarlyStopping to kick in with patience=10).

---

## 6. Inverse Scaling — How Predictions Are Denormalized

The model outputs a value in **[0, 1]** (normalised space). To get actual unit demand:

$$demand_{actual} = demand_{scaled} \times (demand_{max} - demand_{min}) + demand_{min}$$

From our dataset:
- `demand_min` = 0.0 units
- `demand_max` = 408.0 units

So a model output of 0.22 translates to:
- 0.22 × 408 + 0 = **~89.8 units of demand**

---

## 7. Evaluation Metrics

```python
# Computed on test set (chronologically last 15% of data)
overall_mae  = 9.6245   units
overall_rmse = 15.3808  units
overall_r2   = 0.9611
```

### MAE — Mean Absolute Error
$$MAE = \frac{1}{n} \sum |y_{pred} - y_{true}|$$

**Result: 9.62 units** — On average, the model's daily demand forecast is off by ±9.62 units. Given `avg_demand = 90.17 units`, this is a **~10.7% average error**.

### RMSE — Root Mean Square Error
$$RMSE = \sqrt{\frac{1}{n} \sum (y_{pred} - y_{true})^2}$$

**Result: 15.38 units** — Higher than MAE because RMSE penalises large errors more. The gap between MAE and RMSE (9.62 vs 15.38) tells us there are occasional larger misses (the model isn't consistently wrong by a little — it has some big outlier errors mixed in).

### R² Score (Coefficient of Determination)
$$R² = 1 - \frac{\sum (y_{true} - y_{pred})^2}{\sum (y_{true} - \bar{y})^2}$$

**Result: 0.9611** — The model explains **96.11%** of the variance in demand. 

- R² = 1.0 → Perfect prediction  
- R² = 0.0 → Model is no better than predicting the mean demand every day  
- R² = 0.9611 → Excellent. Only 3.89% of variability is unexplained.

---

## 8. Per-Entity Performance

| Entity | MAE (units) | Difficulty Explanation |
|---|---|---|
| Store_B - Furniture | **2.62** ← Best | Low/stable demand, predictable |
| Store_C - Furniture | 3.12 | Similar pattern |
| Store_A - Furniture | 4.82 | Slightly more volatile |
| Store_B - Electronics | 6.55 | Moderate variance |
| Store_C - Electronics | 5.36 | Moderate |
| Store_A - Electronics | 6.71 | Moderate |
| Store_B - Toys | 6.94 | Seasonal spikes |
| Store_A - Toys | 7.28 | Seasonal |
| Store_A - Clothing | 8.94 | Fashion volatility |
| Store_C - Toys | 8.34 | Seasonal + volatile |
| Store_B - Clothing | 11.17 | High variance |
| Store_C - Clothing | 11.28 | High variance |
| Store_A - Groceries | 20.27 | High volume, daily swings |
| Store_C - Groceries | 20.09 | High volume, daily swings |
| Store_B - Groceries | **20.87** ← Hardest | Highest absolute demand |

**Key insight:** Groceries have the highest MAE in absolute terms because they have the highest daily demand volumes. As a **percentage** of average demand, Furniture (low-volume) and Groceries (high-volume) likely have similar relative accuracy.

---

## 9. Business Impact: Inventory Cost Reduction

With the model's accuracy (MAE = ±9.62 units):

**Naive ordering (without model):**
A safety stock of ~2×MAE = ~19 extra units ordered daily per entity to cover uncertainty.
- Holding cost = 19 units × ₹1.64/day = **₹31.16/day/entity**

**AI-powered ordering:**
Safety stock drops to ~0.5×MAE = ~5 extra units (tighter confidence band).
- Holding cost = 5 units × ₹1.64/day = **₹8.20/day/entity**

**Savings:** ₹22.96/day × 15 entities = **₹344/day total holding cost reduction**

For stockout prevention:
The ±9.62 MAE means the model can signal when to pre-order before stockouts, reducing stockout events at ₹12.30/unit penalty avoided.

---

## 10. Training Observations (From Actual Logs)

```
Epoch  1: loss=0.0127, val_loss=0.0044 (LR=0.001)
Epoch  5: loss=0.0030, val_loss=0.0021  ← Big early improvement
Epoch 11: loss=0.0028, val_loss=0.0016  ← Saved as best here (approx)
Epoch 17: loss=0.0016, val_loss=0.0019  ← LR reduced 0.001→0.0005
Epoch 25: loss=0.0014, val_loss=0.0013  ← LR reduced →0.00025
Epoch 33: loss=0.0011, val_loss=0.0009  ← Best model around here
Epoch 50: loss=0.0010, val_loss=0.0009  (LR=0.000031)
```

**Observations:**
- Training loss dropped from 0.0127 → 0.0010 (92% reduction)
- Validation loss closely tracks training loss → no overfitting
- The learning rate annealing allowed continued improvement even in late epochs
- The small gap between train_loss and val_loss throughout confirms the Dropout layers successfully prevented overfitting

---

## 11. Reproducibility Seeds

```python
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

All random processes (weight initialisation, Dropout, data shuffling) are seeded so the experiment produces the same results every run. This is essential for scientific validity.

---

## 12. Website: How It Works

The website (`project/website/`) is a **static HTML+JS dashboard** that reads pre-computed JSON files output by the model.

### Data flow:
```
Colab (Python + TensorFlow)
     ↓ trained model makes predictions
     ↓ export_artifacts.py runs
     ↓
3 JSON files generated:
  model_metadata.json   → Overall KPIs (MAE, RMSE, R², per-entity MAE)
  forecasts.json        → For each of 15 entities: dates[], actual[], predicted[]
  training_history.json → loss[], val_loss[] per epoch

     ↓ copied to website/ folder
     ↓
Browser (app.js via fetch())
  → Loads JSON files
  → Populates KPI cards (animated counters)
  → Renders Actual vs Predicted chart (Chart.js)
  → Renders Training History loss curve
  → Renders Per-Entity MAE bar list
  → Calculates and shows Inventory Cost Impact panel
```

### Why not run the model in the browser?
TensorFlow/Keras `.keras` models require Python + TensorFlow runtime. Browsers can only run JavaScript. We'd need TensorFlow.js (a separate conversion step) for live inference. The current design pre-bakes all predictions into JSON for simplicity — this is a common production pattern called **batch inference**.

---

## 13. Key Technical Decisions — Summary

| Decision | What | Why |
|---|---|---|
| **Bi-LSTM over LSTM** | Two directional passes | Captures both past and future context within lookback window |
| **30-day lookback** | 30 timesteps history | Balances context richness vs computational cost |
| **Entity-aware sequences** | No cross-entity mixing | Maintains temporal integrity per store-product pair |
| **OHE for categoricals** | Binary indicator columns | Encodes store/product identity without ordinal bias |
| **Scaler fitted on train-only** | No test data in scaler | Prevents data leakage |
| **Chronological split** | 70/15/15 by date | Prevents future information leaking into training |
| **EarlyStopping** | patience=10 | Stops training when improvement plateaus |
| **ReduceLROnPlateau** | factor=0.5, patience=5 | Fine-tunes weights as convergence approaches |
| **ModelCheckpoint** | save_best_only=True | Preserves globally best model, not just last |
| **MSE loss** | Squared error | Penalises large errors more, important in inventory |
| **Dropout(0.2)** | 20% neuron drop | Regularisation, prevents overfitting |
| **seed=42 everywhere** | Reproducibility | Scientific consistency across runs |
