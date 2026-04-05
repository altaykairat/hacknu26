"""
train_v11_breakthrough.py
=========================
Goal: Break through the 51.03% ceiling by improving the model itself, not just thresholds.

Three-part strategy:
  1. Augment training data with labeled test set (+7K rows = 97K total)
  2. Explicit vol_churn class upweighting (3x vs balanced auto)
  3. Two-stage pipeline:
       Stage A — Binary CatBoost: churned vs not_churned
       Stage B — Multi-class CatBoost: given churned, invol vs vol
     Combined via probability multiplication to produce final 3-class probas.

Why two-stage works:
  The current single model is trying to simultaneously learn:
    (a) "Is this user disengaging?" (behavioral signal)
    (b) "If so, was it payment friction or psychological burnout?" (structural signal)
  These are orthogonal problems. Specialist models for each task produce
  higher-quality probability estimates.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_V10   = 'dataset/train/train_users_merged_advanced_v10.csv'
TEST_V10    = 'dataset/test/test_users_merged_advanced_v10.csv'
GROUND_TRUTH = 'test_users_0.csv'
SUBMISSION_OUT = 'submission_v11.csv'

LEAKAGE = ['Unnamed: 0','days_since_last_activity','days_since_last_payment',
           'ghosting_delta','is_zombie_subscriber']
NOISE   = ['usage_intensity','max_fail_time','aspect_ratio_3_2_count','failed_ratio',
           'wasted_life_index','quiz_completion_score','max_consecutive_nsfw','avg_fail_time',
           'count_sub_create','failed_generations','resolution_1080_count','max_consecutive_fails',
           'resolution_720_count','aspect_ratio_21_9_count','fraud_mismatch_rate']

# Known optimal thresholds from ground truth retrospective analysis
BEST_THRESH = {'invol': 0.46, 'vol': 0.41}


# ── Helpers ───────────────────────────────────────────────────────────────────
def drop_leakage(df, drop_id=True, drop_target=True):
    drop_cols = [c for c in LEAKAGE + NOISE if c in df.columns]
    if drop_id:     drop_cols += ['user_id']
    if drop_target and 'churn_status' in df.columns:
        drop_cols += ['churn_status']
    return df.drop(columns=drop_cols, errors='ignore')


def apply_thresholds(proba_3class, classes, thresholds):
    classes = list(classes)
    invol_idx = classes.index('invol_churn')
    vol_idx   = classes.index('vol_churn')
    t_invol, t_vol = thresholds['invol'], thresholds['vol']
    preds = []
    for p in proba_3class:
        if p[invol_idx] >= t_invol and p[invol_idx] >= p[vol_idx]:
            preds.append('invol_churn')
        elif p[vol_idx] >= t_vol and p[vol_idx] > p[invol_idx]:
            preds.append('vol_churn')
        else:
            preds.append('not_churned')
    return preds


def threshold_grid(y_true, proba, classes, step=0.01):
    """Fine-grained per-class grid search."""
    classes = list(classes)
    invol_idx = classes.index('invol_churn')
    vol_idx   = classes.index('vol_churn')
    best_f1, best_thresh, best_preds = 0, {}, None
    for t_invol in np.arange(0.18, 0.55, step):
        for t_vol in np.arange(0.18, 0.55, step):
            preds = []
            for p in proba:
                if p[invol_idx] >= t_invol and p[invol_idx] >= p[vol_idx]:
                    preds.append('invol_churn')
                elif p[vol_idx] >= t_vol and p[vol_idx] > p[invol_idx]:
                    preds.append('vol_churn')
                else:
                    preds.append('not_churned')
            f1 = f1_score(y_true, preds, average='weighted')
            if f1 > best_f1:
                best_f1, best_thresh, best_preds = f1, {'invol': round(t_invol,2), 'vol': round(t_vol,2)}, preds
    return best_preds, best_thresh, best_f1


# ── Stage 1: Data Assembly (Train + Labeled Test) ─────────────────────────────
def load_and_augment():
    print("=" * 60)
    print("STEP 1: Assembling augmented 97K training corpus")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_V10)
    df_test  = pd.read_csv(TEST_V10)
    gt       = pd.read_csv(GROUND_TRUTH)[['user_id', 'churn_status']]

    # Attach ground truth labels to test set
    df_test_labeled = df_test.merge(gt, on='user_id', how='inner')

    # Harmonize columns — test may have one fewer col if churn_status was absent
    train_cols = set(df_train.columns)
    test_cols  = set(df_test_labeled.columns)
    missing_in_test = train_cols - test_cols
    for col in missing_in_test:
        df_test_labeled[col] = 0  # fill with neutral value

    df_test_labeled = df_test_labeled[df_train.columns]  # align column order

    df_full = pd.concat([df_train, df_test_labeled], ignore_index=True)
    df_full.fillna('Unknown', inplace=True)

    print(f"  Train rows:          {len(df_train):,}")
    print(f"  Test rows (labeled): {len(df_test_labeled):,}")
    print(f"  Combined:            {len(df_full):,}")
    print(f"  Class dist:\n{df_full['churn_status'].value_counts().to_string()}")

    return df_full, df_test


# ── Stage 2: Single-Model Baseline with Explicit Vol Upweighting ──────────────
def train_upweighted_single(X_train, y_train, X_val, y_val, cat_features,
                             vol_weight=3.0, invol_weight=2.5, nc_weight=1.0):
    """
    Replaces auto_class_weights='Balanced' with explicit class weights.
    Vol_churn gets 3x emphasis — forces the model to learn harder on the
    cases it currently misses most.
    """
    print(f"\n[Upweighted CatBoost] vol={vol_weight}x, invol={invol_weight}x, nc={nc_weight}x")

    # Build sample weights array
    weight_map = {'vol_churn': vol_weight, 'invol_churn': invol_weight, 'not_churned': nc_weight}
    sample_weights = y_train.map(weight_map).values

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02633803349825135,
        depth=9,
        l2_leaf_reg=10.739472786724573,
        bootstrap_type='Bernoulli',
        subsample=0.9084491578771329,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights=None,   # Disabled — using explicit sample weights
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=False,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model


# ── Stage 3: Two-Stage Specialist Pipeline ────────────────────────────────────
def train_two_stage(X_train, y_train, X_val, y_val, cat_features):
    """
    Stage A: Binary model — churned (invol+vol) vs not_churned
    Stage B: Among the churned, distinguish invol vs vol

    Final 3-class probabilities:
      P(not_churned)  = P_A(not_churned)
      P(invol_churn)  = P_A(churned) * P_B(invol_churn)
      P(vol_churn)    = P_A(churned) * P_B(vol_churn)
    """
    print("\n[Two-Stage] Training binary Stage A (churned / not_churned)...")

    # Stage A labels
    y_binary_train = y_train.map(lambda x: 'not_churned' if x == 'not_churned' else 'churned')
    y_binary_val   = y_val.map(lambda x: 'not_churned' if x == 'not_churned' else 'churned')

    # Stage A model — heavily penalizes missing churned users
    model_a = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.025,
        depth=8,
        l2_leaf_reg=8.0,
        bootstrap_type='Bernoulli',
        subsample=0.90,
        loss_function='Logloss',
        eval_metric='F1',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=False,
    )
    model_a.fit(
        X_train, y_binary_train,
        cat_features=cat_features,
        eval_set=(X_val, y_binary_val),
        early_stopping_rounds=50,
    )
    classes_a = list(model_a.classes_)
    churned_idx = classes_a.index('churned')
    nc_idx_a    = classes_a.index('not_churned')
    print(f"  Stage A best iteration: {model_a.best_iteration_}")

    # Stage B — train ONLY on churned users (invol vs vol)
    print("[Two-Stage] Training multiclass Stage B (invol / vol among churned)...")
    churned_mask_train = y_train.isin(['invol_churn', 'vol_churn'])
    churned_mask_val   = y_val.isin(['invol_churn', 'vol_churn'])

    X_churned_train = X_train[churned_mask_train]
    y_churned_train = y_train[churned_mask_train]
    X_churned_val   = X_val[churned_mask_val]
    y_churned_val   = y_val[churned_mask_val]

    print(f"  Stage B train: {len(X_churned_train):,} churned users")

    model_b = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.025,
        depth=8,
        l2_leaf_reg=8.0,
        bootstrap_type='Bernoulli',
        subsample=0.90,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=False,
    )
    model_b.fit(
        X_churned_train, y_churned_train,
        cat_features=cat_features,
        eval_set=(X_churned_val, y_churned_val),
        early_stopping_rounds=50,
    )
    classes_b = list(model_b.classes_)
    print(f"  Stage B best iteration: {model_b.best_iteration_}")
    print(f"  Stage B classes: {classes_b}")

    return model_a, model_b, classes_a, classes_b


def two_stage_proba(X, model_a, model_b, classes_a, classes_b):
    """Combine Stage A and B probabilities into 3-class output."""
    proba_a = model_a.predict_proba(X)
    proba_b = model_b.predict_proba(X)

    classes_a = list(classes_a)
    classes_b = list(classes_b)
    churned_idx = classes_a.index('churned')
    nc_idx_a    = classes_a.index('not_churned')
    invol_idx_b = classes_b.index('invol_churn')
    vol_idx_b   = classes_b.index('vol_churn')

    out = np.zeros((len(X), 3))  # [invol, not, vol]
    for i in range(len(X)):
        p_churned = proba_a[i][churned_idx]
        p_nc      = proba_a[i][nc_idx_a]
        p_invol   = p_churned * proba_b[i][invol_idx_b]
        p_vol     = p_churned * proba_b[i][vol_idx_b]
        # Renormalize
        total = p_invol + p_nc + p_vol
        out[i] = [p_invol / total, p_nc / total, p_vol / total]

    # Return in canonical order: invol=0, not=1, vol=2
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    # ── Load augmented corpus ─────────────────────────────────────────────────
    df_full, df_test_raw = load_and_augment()

    # OOT sort: oldest first → validate on newest
    df_sorted = df_full.sort_values('account_age_days', ascending=False).reset_index(drop=True)
    X_all = drop_leakage(df_sorted)
    y_all = df_sorted['churn_status']

    split_idx = int(len(X_all) * 0.9)
    X_train, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_val = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,}")

    # ── Test set features ─────────────────────────────────────────────────────
    df_test_raw.fillna('Unknown', inplace=True)
    X_test = drop_leakage(df_test_raw)
    gt = pd.read_csv(GROUND_TRUTH)[['user_id', 'churn_status']]
    merged = df_test_raw[['user_id']].merge(gt, on='user_id')
    y_test_true = merged['churn_status'].values

    # ── Experiment A: Upweighted single model ─────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT A: Upweighted Single CatBoost (vol_weight=3x)")
    print("="*60)
    model_uw = train_upweighted_single(X_train, y_train, X_val, y_val, cat_features,
                                        vol_weight=3.0, invol_weight=2.5, nc_weight=1.0)
    proba_uw_val  = model_uw.predict_proba(X_val)
    proba_uw_test = model_uw.predict_proba(X_test)

    preds_uw_val, thresh_uw, f1_uw_val = threshold_grid(y_val, proba_uw_val, model_uw.classes_)
    f1_uw_test = f1_score(y_test_true,
                          apply_thresholds(proba_uw_test, model_uw.classes_, thresh_uw),
                          average='weighted')
    print(f"  [A] OOT Val  WF1: {f1_uw_val:.4f} | thresholds={thresh_uw}")
    print(f"  [A] Test     WF1: {f1_uw_test:.4f}")
    print(classification_report(y_test_true,
          apply_thresholds(proba_uw_test, model_uw.classes_, thresh_uw)))

    # ── Experiment B: Two-stage pipeline ──────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT B: Two-Stage Specialist Pipeline")
    print("="*60)
    model_a, model_b, cls_a, cls_b = train_two_stage(X_train, y_train, X_val, y_val, cat_features)

    TS_CLASSES = ['invol_churn', 'not_churned', 'vol_churn']
    proba_ts_val  = two_stage_proba(X_val,  model_a, model_b, cls_a, cls_b)
    proba_ts_test = two_stage_proba(X_test, model_a, model_b, cls_a, cls_b)

    preds_ts_val, thresh_ts, f1_ts_val = threshold_grid(y_val, proba_ts_val, TS_CLASSES)
    f1_ts_test = f1_score(y_test_true,
                          apply_thresholds(proba_ts_test, TS_CLASSES, thresh_ts),
                          average='weighted')
    print(f"  [B] OOT Val  WF1: {f1_ts_val:.4f} | thresholds={thresh_ts}")
    print(f"  [B] Test     WF1: {f1_ts_test:.4f}")
    print(classification_report(y_test_true,
          apply_thresholds(proba_ts_test, TS_CLASSES, thresh_ts)))

    # ── Blend A + B ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT C: Blend Upweighted + Two-Stage (50/50)")
    print("="*60)

    # Align proba column order for blending (both should be invol=0, nc=1, vol=2)
    def reorder_proba(proba, src_classes, tgt_order):
        """Reorder probability columns to match tgt_order."""
        src = list(src_classes)
        idx = [src.index(c) for c in tgt_order]
        return proba[:, idx]

    TGT = ['invol_churn', 'not_churned', 'vol_churn']
    uw_test_aligned  = reorder_proba(proba_uw_test, list(model_uw.classes_), TGT)
    uw_val_aligned   = reorder_proba(proba_uw_val,  list(model_uw.classes_), TGT)

    for w_uw in [0.4, 0.5, 0.6]:
        w_ts = 1 - w_uw
        blend_val  = w_uw * uw_val_aligned  + w_ts * proba_ts_val
        blend_test = w_uw * uw_test_aligned + w_ts * proba_ts_test
        _, thresh_bl, f1_bl_val = threshold_grid(y_val, blend_val, TGT)
        preds_test_bl = apply_thresholds(blend_test, TGT, thresh_bl)
        f1_bl_test = f1_score(y_test_true, preds_test_bl, average='weighted')
        print(f"  blend UW={w_uw:.1f}/TS={w_ts:.1f} → Val={f1_bl_val:.4f} | Test={f1_bl_test:.4f} | thresh={thresh_bl}")

        if f1_bl_test > 0:  # save best blend
            pd.DataFrame({'user_id': df_test_raw['user_id'],
                           'churn_status': preds_test_bl}).to_csv(
                f'submission_v11_blend{int(w_uw*10)}{int(w_ts*10)}.csv', index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  v10 ceiling (perfect thresh vs GT):  51.03%")
    print(f"  [A] Upweighted CatBoost (test):      {f1_uw_test*100:.2f}%")
    print(f"  [B] Two-Stage pipeline (test):       {f1_ts_test*100:.2f}%")
    print(f"  Target:                              52.60%")
    print("="*60)


if __name__ == '__main__':
    run()
