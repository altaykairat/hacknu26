"""
train_ensemble_v10.py
=====================
Ensemble pipeline targeting Weighted F1 > 52%.

Strategy:
  1. Load v10 dataset (v9 + 5 new behavioral features).
  2. OOT 90/10 split (oldest cohorts → train, newest → val).
  3. Train CatBoost (kept from v9 Optuna tuning).
  4. Train LightGBM with equivalent regularization profile.
  5. Per-class threshold grid: independent thresholds for invol_churn / vol_churn.
  6. Blend probabilities: 65% CatBoost + 35% LightGBM.
  7. Re-apply per-class threshold on blended probas.
  8. Optional pseudo-labeling: high-confidence not_churned from test set.
  9. Final model: retrain CatBoost + LightGBM on 100% data for submission.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Config
# ============================================================
TRAIN_CSV  = 'dataset/train/train_users_merged_advanced_v10.csv'
TEST_CSV   = 'dataset/test/test_users_merged_advanced_v10.csv'
CB_MODEL_OUT  = 'catboost_churn_v10.cbm'
SUBMISSION_OUT = 'submission_v10.csv'

LEAKAGE_FEATURES = [
    'Unnamed: 0',
    'days_since_last_activity',
    'days_since_last_payment',
    'ghosting_delta',
    'is_zombie_subscriber',
]
NOISE_FEATURES = [
    'usage_intensity', 'max_fail_time', 'aspect_ratio_3_2_count', 'failed_ratio',
    'wasted_life_index', 'quiz_completion_score', 'max_consecutive_nsfw', 'avg_fail_time',
    'count_sub_create', 'failed_generations', 'resolution_1080_count', 'max_consecutive_fails',
    'resolution_720_count', 'aspect_ratio_21_9_count', 'fraud_mismatch_rate'
]


# ============================================================
# Threshold Tuning — Per-Class Independent Grid
# ============================================================
def tune_per_class_thresholds(y_true, y_proba, classes):
    """
    Independently searches the best threshold for invol_churn and vol_churn.
    For each combination of (t_invol, t_vol), the majority class wins only
    when neither churn class clears its own threshold.

    Returns: (best_preds, best_thresholds_dict, best_f1)
    """
    classes = list(classes)
    majority_idx  = classes.index('not_churned')
    invol_idx     = classes.index('invol_churn')
    vol_idx       = classes.index('vol_churn')

    best_f1     = 0.0
    best_preds  = None
    best_thresh = {}

    print("\n[Threshold Grid] Scanning per-class thresholds...")

    for t_invol in np.arange(0.18, 0.52, 0.02):
        for t_vol in np.arange(0.18, 0.52, 0.02):
            preds = []
            for probas in y_proba:
                p_invol = probas[invol_idx]
                p_vol   = probas[vol_idx]

                if p_invol >= t_invol and p_invol >= p_vol:
                    preds.append('invol_churn')
                elif p_vol >= t_vol and p_vol > p_invol:
                    preds.append('vol_churn')
                else:
                    preds.append('not_churned')

            f1 = f1_score(y_true, preds, average='weighted')
            if f1 > best_f1:
                best_f1    = f1
                best_preds = preds
                best_thresh = {'invol': round(t_invol, 2), 'vol': round(t_vol, 2)}

    print(f"  --> Best thresholds: invol={best_thresh['invol']:.2f}, vol={best_thresh['vol']:.2f}")
    print(f"  --> Weighted F1 with per-class thresholds: {best_f1:.4f}")
    return best_preds, best_thresh, best_f1


def apply_per_class_thresholds(y_proba, classes, thresholds):
    """Apply a fixed per-class threshold dict to a proba array."""
    classes = list(classes)
    majority_idx = classes.index('not_churned')
    invol_idx    = classes.index('invol_churn')
    vol_idx      = classes.index('vol_churn')

    t_invol = thresholds['invol']
    t_vol   = thresholds['vol']

    preds = []
    for probas in y_proba:
        p_invol = probas[invol_idx]
        p_vol   = probas[vol_idx]
        if p_invol >= t_invol and p_invol >= p_vol:
            preds.append('invol_churn')
        elif p_vol >= t_vol and p_vol > p_invol:
            preds.append('vol_churn')
        else:
            preds.append('not_churned')
    return preds


# ============================================================
# Data Loading
# ============================================================
def load_data(csv_path, has_target=True):
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    df.fillna('Unknown', inplace=True)

    drop_cols = [c for c in (LEAKAGE_FEATURES + NOISE_FEATURES) if c in df.columns]

    if has_target:
        X = df.drop(columns=['user_id', 'churn_status'] + drop_cols)
        y = df['churn_status']
        return X, y, df
    else:
        X = df.drop(columns=['user_id'] + drop_cols)
        if 'churn_status' in X.columns:
            X = X.drop(columns=['churn_status'])
        return X, None, df


# ============================================================
# CatBoost Trainer
# ============================================================
def train_catboost(X_train, y_train, X_val, y_val, cat_features, iterations=1500):
    print(f"\n[CatBoost] Training on {len(X_train):,} users...")
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.02633803349825135,
        depth=9,
        l2_leaf_reg=10.739472786724573,
        bootstrap_type='Bernoulli',
        subsample=0.9084491578771329,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=False,
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    print(f"  --> Best iteration: {model.best_iteration_}")
    return model


# ============================================================
# LightGBM Trainer
# ============================================================
def train_lgbm(X_train, y_train, X_val, y_val, cat_features):
    """
    LightGBM needs numeric categoricals — we label-encode them before fitting.
    Returns (model, encoder_map) so we can apply the same encoding at inference.
    """
    print(f"\n[LightGBM] Training on {len(X_train):,} users...")

    # LGB doesn't take raw strings — convert cats to category dtype
    X_tr = X_train.copy()
    X_vl = X_val.copy()
    for col in cat_features:
        combined  = pd.concat([X_tr[col], X_vl[col]]).astype('category')
        cat_dtype = combined.cat
        X_tr[col] = X_tr[col].astype('category').cat.set_categories(cat_dtype.categories).cat.codes
        X_vl[col] = X_vl[col].astype('category').cat.set_categories(cat_dtype.categories).cat.codes

    # Encode target
    label_map  = {'invol_churn': 0, 'not_churned': 1, 'vol_churn': 2}
    y_tr_enc   = y_train.map(label_map)
    y_vl_enc   = y_val.map(label_map)

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.025,
        num_leaves=63,            # ~depth-6 equivalent; leaf-wise growth
        max_depth=-1,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=10.0,          # Mirror CatBoost's heavy L2
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr_enc,
        eval_set=[(X_vl, y_vl_enc)],
        callbacks=[
            __import__('lightgbm').early_stopping(50, verbose=False),
            __import__('lightgbm').log_evaluation(period=-1),
        ],
        categorical_feature=cat_features,
    )
    print(f"  --> Best iteration: {model.best_iteration_}")

    # Store category mappings so test set can be encoded identically
    cat_mappings = {}
    for col in cat_features:
        combined = pd.concat([X_train[col], X_val[col]]).astype('category')
        cat_mappings[col] = combined.cat.categories

    inv_label_map = {0: 'invol_churn', 1: 'not_churned', 2: 'vol_churn'}
    return model, cat_mappings, inv_label_map


def encode_for_lgbm(X, cat_features, cat_mappings):
    X_enc = X.copy()
    for col in cat_features:
        X_enc[col] = X_enc[col].astype('category').cat.set_categories(cat_mappings[col]).cat.codes
    return X_enc


# ============================================================
# Pseudo-Labeling
# ============================================================
def pseudo_label_not_churned(X_test, cb_proba, lgbm_proba, classes,
                               blend_cb=0.65, confidence_threshold=0.90):
    """
    Find test samples where the blended model is >90% confident they are NOT churned.
    Returns the pseudo-labeled subset as (X_pseudo, y_pseudo).
    """
    classes = list(classes)
    nc_idx  = classes.index('not_churned')

    blended = blend_cb * cb_proba + (1 - blend_cb) * lgbm_proba
    nc_conf = blended[:, nc_idx]

    mask = nc_conf >= confidence_threshold
    n = mask.sum()
    print(f"\n[Pseudo-Label] High-confidence not_churned: {n:,} / {len(X_test):,} test users")

    X_pseudo = X_test[mask].copy()
    y_pseudo = pd.Series(['not_churned'] * n, index=X_pseudo.index)
    return X_pseudo, y_pseudo


# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline(use_pseudo_labeling=True, pseudo_confidence=0.92,
                 cb_blend_weight=0.65):

    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    X_all, y_all, df_all = load_data(TRAIN_CSV, has_target=True)
    X_test, _, df_test   = load_data(TEST_CSV,  has_target=False)

    # OOT Sort: oldest cohort first → train on past, validate on future
    df_all_sorted = df_all.sort_values('account_age_days', ascending=False).reset_index(drop=True)
    drop_cols = [c for c in (LEAKAGE_FEATURES + NOISE_FEATURES) if c in df_all_sorted.columns]
    X_all_s = df_all_sorted.drop(columns=['user_id', 'churn_status'] + drop_cols)
    y_all_s = df_all_sorted['churn_status']

    split_idx   = int(len(X_all_s) * 0.9)
    X_train, X_val = X_all_s.iloc[:split_idx], X_all_s.iloc[split_idx:]
    y_train, y_val = y_all_s.iloc[:split_idx], y_all_s.iloc[split_idx:]

    print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Train class dist:\n{y_train.value_counts().to_string()}")

    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical features: {len(cat_features)}")

    # --------------------------------------------------------
    # 2. Train Phase 1: CatBoost
    # --------------------------------------------------------
    cb_model = train_catboost(X_train, y_train, X_val, y_val, cat_features, iterations=1500)
    cb_proba_val = cb_model.predict_proba(X_val)
    classes = cb_model.classes_

    # --------------------------------------------------------
    # 3. Train Phase 2: LightGBM
    # --------------------------------------------------------
    lgbm_model, cat_map, inv_lmap = train_lgbm(X_train, y_train, X_val, y_val, cat_features)
    X_val_enc = encode_for_lgbm(X_val, cat_features, cat_map)
    lgbm_proba_val_raw = lgbm_model.predict_proba(X_val_enc)
    # Reorder LGB columns to match CatBoost class order: invol=0, not=1, vol=2
    lgbm_proba_val = lgbm_proba_val_raw  # already [invol, not_churned, vol] by label_map

    # --------------------------------------------------------
    # 4. Blend Probabilities
    # --------------------------------------------------------
    print(f"\n[Blend] {cb_blend_weight:.0%} CatBoost + {1-cb_blend_weight:.0%} LightGBM")
    blended_val = cb_blend_weight * cb_proba_val + (1 - cb_blend_weight) * lgbm_proba_val

    # --------------------------------------------------------
    # 5. Baseline: shared threshold (legacy)
    # --------------------------------------------------------
    majority_idx = list(classes).index('not_churned')
    def shared_thresh_preds(proba, t=0.42):
        preds = []
        for p in proba:
            churn_p = [prob for i, prob in enumerate(p) if i != majority_idx]
            max_cp  = max(churn_p)
            max_ci  = list(p).index(max_cp)
            preds.append(classes[max_ci] if max_cp >= t else classes[majority_idx])
        return preds

    baseline_preds = shared_thresh_preds(blended_val, t=0.42)
    baseline_f1 = f1_score(y_val, baseline_preds, average='weighted')
    print(f"\n[Baseline Blend @ 0.42] Weighted F1: {baseline_f1:.4f}")

    # --------------------------------------------------------
    # 6. Per-Class Threshold Optimization on blended probas
    # --------------------------------------------------------
    opt_preds, best_thresh, opt_f1 = tune_per_class_thresholds(y_val, blended_val, classes)
    print(f"\n[Optimized Per-Class Threshold] Weighted F1: {opt_f1:.4f}")
    print(classification_report(y_val, opt_preds))

    # --------------------------------------------------------
    # 7. Optional: Pseudo-Labeling on Test Set
    # --------------------------------------------------------
    X_train_final = X_train.copy()
    y_train_final = y_train.copy()

    if use_pseudo_labeling:
        X_test_enc     = encode_for_lgbm(X_test, cat_features, cat_map)
        cb_proba_test  = cb_model.predict_proba(X_test)
        lgbm_proba_test= lgbm_model.predict_proba(X_test_enc)
        blended_test   = cb_blend_weight * cb_proba_test + (1 - cb_blend_weight) * lgbm_proba_test

        X_pseudo, y_pseudo = pseudo_label_not_churned(
            X_test, cb_proba_test, lgbm_proba_test, classes,
            blend_cb=cb_blend_weight, confidence_threshold=pseudo_confidence
        )

        if len(X_pseudo) > 0:
            # Align pseudo features with training feature set
            X_pseudo = X_pseudo[X_train_final.columns]
            X_train_final = pd.concat([X_train_final, X_pseudo], ignore_index=True)
            y_train_final = pd.concat([y_train_final, y_pseudo], ignore_index=True)
            print(f"  Augmented train set: {len(X_train_final):,} rows")

    # --------------------------------------------------------
    # 8. Final Full-Data Retraining for Submission
    # --------------------------------------------------------
    # Use ALL training data (+ pseudo labels if enabled), no val split
    print("\n[Final] Retraining on FULL dataset for submission...")
    X_full = pd.concat([X_all_s, X_train_final.iloc[len(X_train):]], ignore_index=True) \
             if use_pseudo_labeling and len(X_pseudo) > 0 else X_all_s
    y_full = pd.concat([y_all_s, y_train_final.iloc[len(y_train):]], ignore_index=True) \
             if use_pseudo_labeling and len(X_pseudo) > 0 else y_all_s

    cat_full = X_full.select_dtypes(include=['object']).columns.tolist()

    final_cb = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02633803349825135,
        depth=9,
        l2_leaf_reg=10.739472786724573,
        bootstrap_type='Bernoulli',
        subsample=0.9084491578771329,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=100,
    )
    final_cb.fit(X_full, y_full, cat_features=cat_full)
    final_cb.save_model(CB_MODEL_OUT)
    print(f"  CatBoost final model saved → {CB_MODEL_OUT}")

    # Retrain LightGBM on full data
    lgbm_full, cat_map_full, _ = train_lgbm(X_full, y_full, X_val, y_val, cat_full)

    # --------------------------------------------------------
    # 9. Inference on Test Set → Submission
    # --------------------------------------------------------
    print("\n[Inference] Generating test predictions...")
    cb_proba_test_final  = final_cb.predict_proba(X_test)
    X_test_enc_final     = encode_for_lgbm(X_test, cat_full, cat_map_full)
    lgbm_proba_test_final= lgbm_full.predict_proba(X_test_enc_final)

    blended_test_final = (cb_blend_weight * cb_proba_test_final
                          + (1 - cb_blend_weight) * lgbm_proba_test_final)

    final_preds = apply_per_class_thresholds(blended_test_final, final_cb.classes_, best_thresh)

    submission = pd.DataFrame({
        'user_id': df_test['user_id'],
        'churn_status': final_preds,
    })
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"  Submission saved → {SUBMISSION_OUT}")
    print(f"  Prediction distribution:\n{submission['churn_status'].value_counts().to_string()}")

    # --------------------------------------------------------
    # 10. Summary
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("ENSEMBLE PIPELINE SUMMARY")
    print("="*60)
    print(f"  Baseline (CatBoost only, thresh=0.42):         {baseline_f1:.4f}")
    print(f"  Blended + Per-Class Thresholds (OOT Val):      {opt_f1:.4f}")
    print(f"  Best thresholds → invol={best_thresh['invol']}, vol={best_thresh['vol']}")
    print(f"  Pseudo-labeled examples added: {len(X_pseudo) if use_pseudo_labeling else 0}")
    print(f"  Submission: {SUBMISSION_OUT}")
    print("="*60)

    return opt_f1, best_thresh


if __name__ == '__main__':
    run_pipeline(
        use_pseudo_labeling=True,
        pseudo_confidence=0.92,   # Only >92% confident not_churned → augment
        cb_blend_weight=0.65,     # 65% CatBoost, 35% LightGBM
    )
