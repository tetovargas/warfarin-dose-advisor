"""
warfarin_ml_pipeline.py
=======================
Dose Recommendation Engine for Mechanical Mitral Valve Patients
Compares: Random Forest | XGBoost | LSTM (sequential)
Target: New_Warfarin_Dose_mg_day (regression)
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import json

np.random.seed(42)
tf.random.set_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  WARFARIN DOSE RECOMMENDATION ENGINE — ML PIPELINE")
print("=" * 65)

demo_df  = pd.read_excel('warfarin_cohort_MMV.xlsx', sheet_name='Patient Demographics',   skiprows=2)
weekly_df = pd.read_excel('warfarin_cohort_MMV.xlsx', sheet_name='Weekly INR Records',    skiprows=2)

# Normalise column names
demo_df.columns   = [c.strip().replace(' ', '_') for c in demo_df.columns]
weekly_df.columns = [c.strip().replace(' ', '_') for c in weekly_df.columns]

print(f"\n[DATA]  Patients: {demo_df['Patient_ID'].nunique()}  |  Weekly rows: {len(weekly_df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

# --- 2a. Patient-level features ---
demo_feats = demo_df[['Patient_ID','Age','Sex','Weight_kg','Height_cm','BMI',
                       'CYP2C9_Genotype','VKORC1_Genotype','Comorbidities']].copy()

# Binary comorbidity flags
COMORBIDITIES = ['Atrial_Fibrillation','Heart_Failure','Hypertension',
                 'Diabetes_Mellitus','CKD','Hypothyroidism','Hyperthyroidism',
                 'Liver_Disease','COPD','Coronary_Artery_Disease']
for co in COMORBIDITIES:
    key = co.replace('_', ' ').replace('CKD', 'CKD').lower()
    demo_feats[f'has_{co}'] = demo_feats['Comorbidities'].str.lower().str.contains(
        key.replace('_',' '), na=False).astype(int)

# Encode genotypes
cyp_map   = {'*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*2': 3, '*2/*3': 4, '*3/*3': 5}
vkorc_map = {'GG': 0, 'AG': 1, 'AA': 2}
sex_map   = {'M': 0, 'F': 1}
demo_feats['CYP2C9_enc']  = demo_feats['CYP2C9_Genotype'].map(cyp_map).fillna(0)
demo_feats['VKORC1_enc']  = demo_feats['VKORC1_Genotype'].map(vkorc_map).fillna(1)
demo_feats['Sex_enc']     = demo_feats['Sex'].map(sex_map).fillna(0)

# --- 2b. Weekly-level features ---
vitk_map     = {'Low': 0, 'Normal': 1, 'High': 2}
illness_map  = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
exercise_map = {'Sedentary': 0, 'Moderate': 1, 'Active': 2}
yn_map       = {'Yes': 1, 'No': 0}
med_eff_map  = {'None': 0, 'Increase': 1, 'Decrease': -1, 'Mild Inc': 1}

# Drug category encoding
drug_cats = weekly_df['Medication_Category'].fillna('None').unique()
drug_cat_enc = {v: i for i, v in enumerate(sorted(drug_cats))}

weekly_df['VitK_enc']        = weekly_df['VitK_Intake_Level'].map(vitk_map).fillna(1)
weekly_df['Illness_enc']     = weekly_df['Illness_Event'].map(illness_map).fillna(0)
weekly_df['Exercise_enc']    = weekly_df['Exercise_Level'].map(exercise_map).fillna(1)
weekly_df['Cranberry_enc']   = weekly_df['Cranberry_Juice'].map(yn_map).fillna(0)
weekly_df['Grapefruit_enc']  = weekly_df['Grapefruit'].map(yn_map).fillna(0)
weekly_df['Garlic_enc']      = weekly_df['Garlic_Supplement'].map(yn_map).fillna(0)
weekly_df['Diarrhea_enc']    = weekly_df['Diarrhea_Vomiting'].map(yn_map).fillna(0)
weekly_df['MedEffect_enc']   = weekly_df['Medication_INR_Effect'].map(med_eff_map).fillna(0)
weekly_df['DrugCat_enc']     = weekly_df['Medication_Category'].fillna('None').map(drug_cat_enc).fillna(0)

# --- 2c. Lag features (previous 2 weeks' INR and dose) ---
weekly_df = weekly_df.sort_values(['Patient_ID', 'Week_Number'])
weekly_df['INR_lag1']  = weekly_df.groupby('Patient_ID')['INR_Result'].shift(1)
weekly_df['INR_lag2']  = weekly_df.groupby('Patient_ID')['INR_Result'].shift(2)
weekly_df['Dose_lag1'] = weekly_df.groupby('Patient_ID')['Current_Warfarin_Dose_mg_day'].shift(1)
weekly_df['INR_delta'] = weekly_df['INR_Result'] - weekly_df['INR_lag1']  # week-over-week change

# --- 2d. Merge ---
df = weekly_df.merge(demo_feats[['Patient_ID','Age','Sex_enc','Weight_kg','BMI',
                                  'CYP2C9_enc','VKORC1_enc'] +
                                 [f'has_{c}' for c in COMORBIDITIES]],
                     on='Patient_ID', how='left')

# Drop rows with NaN target or lag features
df = df.dropna(subset=['New_Warfarin_Dose_mg_day','INR_lag1'])

print(f"[FEATURES]  Usable rows after lag drop: {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT — by patient (avoid leakage)
# ═══════════════════════════════════════════════════════════════════════════════
all_patients = df['Patient_ID'].unique()
np.random.shuffle(all_patients)
split_idx   = int(len(all_patients) * 0.80)
train_pids  = all_patients[:split_idx]
test_pids   = all_patients[split_idx:]

TABULAR_FEATURES = [
    'INR_Result', 'INR_lag1', 'INR_lag2', 'INR_delta',
    'Current_Warfarin_Dose_mg_day', 'Dose_lag1',
    'VitK_enc', 'Alcohol_Units_Per_Week', 'Cranberry_enc', 'Grapefruit_enc',
    'Garlic_enc', 'MedEffect_enc', 'DrugCat_enc',
    'Missed_Doses_This_Week', 'Illness_enc', 'Diarrhea_enc', 'Exercise_enc',
    'Age', 'Sex_enc', 'Weight_kg', 'BMI',
    'CYP2C9_enc', 'VKORC1_enc',
] + [f'has_{c}' for c in COMORBIDITIES]

TARGET = 'New_Warfarin_Dose_mg_day'

train_df = df[df['Patient_ID'].isin(train_pids)]
test_df  = df[df['Patient_ID'].isin(test_pids)]

X_train = train_df[TABULAR_FEATURES].fillna(0).values
y_train = train_df[TARGET].values
X_test  = test_df[TABULAR_FEATURES].fillna(0).values
y_test  = test_df[TARGET].values

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"[SPLIT]  Train patients: {len(train_pids)}  ({len(X_train):,} rows) | "
      f"Test patients: {len(test_pids)}  ({len(X_test):,} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. METRICS HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5) * 100
    within_10 = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    within_20 = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
    result = {
        'Model': name,
        'RMSE_mg': round(rmse, 4),
        'MAE_mg': round(mae, 4),
        'R2': round(r2, 4),
        'Within_0.5mg_%': round(within_05, 1),
        'Within_1.0mg_%': round(within_10, 1),
        'Within_2.0mg_%': round(within_20, 1),
    }
    print(f"\n  ── {name} ──")
    print(f"     RMSE:           {rmse:.4f} mg/day")
    print(f"     MAE:            {mae:.4f} mg/day")
    print(f"     R²:             {r2:.4f}")
    print(f"     Within ±0.5mg:  {within_05:.1f}%")
    print(f"     Within ±1.0mg:  {within_10:.1f}%")
    print(f"     Within ±2.0mg:  {within_20:.1f}%")
    return result

results = []

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  MODEL 1: Random Forest Regressor")
print("─" * 65)
rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                           n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, rf_preds))

# Feature importance
fi = pd.Series(rf.feature_importances_, index=TABULAR_FEATURES).sort_values(ascending=False)
print("\n  Top 10 features (RF):")
for feat, imp in fi.head(10).items():
    print(f"    {feat:<40} {imp:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  MODEL 2: XGBoost Regressor")
print("─" * 65)
xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0,
                   random_state=42, verbosity=0)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)
xgb_preds = xgb.predict(X_test)
results.append(evaluate("XGBoost", y_test, xgb_preds))

xgb_fi = pd.Series(xgb.feature_importances_, index=TABULAR_FEATURES).sort_values(ascending=False)
print("\n  Top 10 features (XGB):")
for feat, imp in xgb_fi.head(10).items():
    print(f"    {feat:<40} {imp:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. LSTM — SEQUENTIAL MODEL (5-week lookback window)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  MODEL 3: LSTM Sequential (5-week lookback)")
print("─" * 65)

LOOKBACK = 5
SEQ_FEATURES = [
    'INR_Result', 'Current_Warfarin_Dose_mg_day',
    'VitK_enc', 'Alcohol_Units_Per_Week', 'MedEffect_enc',
    'Missed_Doses_This_Week', 'Illness_enc', 'Diarrhea_enc',
    'Cranberry_enc', 'Grapefruit_enc', 'Garlic_enc',
    'Exercise_enc', 'DrugCat_enc'
]
STATIC_FEATURES = [
    'Age', 'Sex_enc', 'Weight_kg', 'BMI', 'CYP2C9_enc', 'VKORC1_enc'
] + [f'has_{c}' for c in COMORBIDITIES]

def build_sequences(patient_df_group, lookback):
    """Build (seq_X, static_X, y) arrays from a patient's sorted weekly records."""
    seq_vals    = patient_df_group[SEQ_FEATURES].fillna(0).values
    static_vals = patient_df_group[STATIC_FEATURES].fillna(0).values
    targets     = patient_df_group[TARGET].values
    Xs, Xst, ys = [], [], []
    for i in range(lookback, len(seq_vals)):
        Xs.append(seq_vals[i-lookback:i])
        Xst.append(static_vals[i])
        ys.append(targets[i])
    return np.array(Xs), np.array(Xst), np.array(ys)

# Build sequences per patient, then split
def make_dataset(pids):
    all_seq, all_static, all_y = [], [], []
    for pid in pids:
        p = df[df['Patient_ID'] == pid].sort_values('Week_Number')
        if len(p) <= LOOKBACK:
            continue
        s, st, y = build_sequences(p, LOOKBACK)
        all_seq.append(s); all_static.append(st); all_y.append(y)
    return (np.concatenate(all_seq),
            np.concatenate(all_static),
            np.concatenate(all_y))

Xs_train, Xst_train, ys_train = make_dataset(train_pids)
Xs_test,  Xst_test,  ys_test  = make_dataset(test_pids)

# Normalise sequences
seq_scaler = StandardScaler()
n_seq  = Xs_train.shape[2]
Xs_train_s = seq_scaler.fit_transform(Xs_train.reshape(-1, n_seq)).reshape(Xs_train.shape)
Xs_test_s  = seq_scaler.transform(Xs_test.reshape(-1, n_seq)).reshape(Xs_test.shape)

static_scaler = StandardScaler()
Xst_train_s   = static_scaler.fit_transform(Xst_train)
Xst_test_s    = static_scaler.transform(Xst_test)

print(f"  LSTM sequences — Train: {Xs_train_s.shape}  |  Test: {Xs_test_s.shape}")

# Dual-input LSTM: sequential + static context
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model

seq_input    = tf.keras.Input(shape=(LOOKBACK, n_seq), name='seq_input')
static_input = tf.keras.Input(shape=(len(STATIC_FEATURES),), name='static_input')

x = LSTM(64, return_sequences=True)(seq_input)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)

s = Dense(16, activation='relu')(static_input)
combined = Concatenate()([x, s])
combined = Dense(32, activation='relu')(combined)
combined = Dense(16, activation='relu')(combined)
output   = Dense(1, activation='linear')(combined)

lstm_model = Model(inputs=[seq_input, static_input], outputs=output)
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
history = lstm_model.fit(
    [Xs_train_s, Xst_train_s], ys_train,
    validation_split=0.15,
    epochs=80,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

lstm_preds = lstm_model.predict([Xs_test_s, Xst_test_s], verbose=0).flatten()
results.append(evaluate("LSTM (5-wk lookback)", ys_test, lstm_preds))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  HEAD-TO-HEAD COMPARISON")
print("=" * 65)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Identify winner per metric
best_rmse = results_df.loc[results_df['RMSE_mg'].idxmin(), 'Model']
best_r2   = results_df.loc[results_df['R2'].idxmax(), 'Model']
best_prec = results_df.loc[results_df['Within_1.0mg_%'].idxmax(), 'Model']

print(f"\n  Best RMSE:           {best_rmse}")
print(f"  Best R²:             {best_r2}")
print(f"  Best ±1 mg accuracy: {best_prec}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
results_df.to_csv('/sessions/focused-determined-mayer/mnt/outputs/model_comparison_results.csv', index=False)

# Save feature importances
fi_df = pd.DataFrame({
    'Feature':    TABULAR_FEATURES,
    'RF_Importance':  rf.feature_importances_,
    'XGB_Importance': xgb.feature_importances_,
}).sort_values('XGB_Importance', ascending=False)
fi_df.to_csv('/sessions/focused-determined-mayer/mnt/outputs/feature_importances.csv', index=False)

print("\n[SAVED]  model_comparison_results.csv")
print("[SAVED]  feature_importances.csv")
print("\nPipeline complete.")
