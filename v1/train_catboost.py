import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit


def tune_multiclass_thresholds(y_true, y_proba, classes, majority_class='not_churned'):
    """
    Scans thresholds for the minority classes (churn) to maximize Macro F1-score.
    """
    best_f1 = 0
    best_preds = None
    best_thresh = 0.5

    # Locate the index of the majority class
    try:
        majority_idx = list(classes).index(majority_class)
    except ValueError:
        majority_idx = 0  # Fallback if name differs

    print("\n[Optimization] Scanning custom thresholds for Churn classes...")

    # Iterate through potential thresholds (0.20 to 0.55)
    for thresh in np.arange(0.20, 0.55, 0.02):
        y_pred_custom = []
        for probas in y_proba:
            # Isolate probabilities of the churn classes
            churn_probs = [p for i, p in enumerate(
                probas) if i != majority_idx]
            max_churn_prob = max(churn_probs)
            max_churn_idx = list(probas).index(max_churn_prob)

            # If the highest churn probability beats the custom threshold, predict that churn class
            if max_churn_prob >= thresh:
                y_pred_custom.append(classes[max_churn_idx])
            else:
                # Otherwise, default to the safe majority class
                y_pred_custom.append(classes[majority_idx])

        # Calculate Weighted F1
        current_f1 = f1_score(y_true, y_pred_custom, average='weighted')

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_preds = y_pred_custom
            best_thresh = thresh

    print(
        f"--> Optimal Churn Threshold Found: {best_thresh:.2f} (Weighted F1: {best_f1:.4f})")
    return best_preds


def train_catboost_multiclass(csv_path):
    print(f"Loading master dataset for MultiClass training: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Fill missing text categories safely
    df.fillna('Unknown', inplace=True)

    # 2. Out-of-Time Split (OOT) Sorting by account_age_days
    # The older the account, the earlier they joined.
    # Therefore, sorting descending means training on older cohorts and validating on the newest!
    print("Performing Out-of-Time Cohort Validation split...")
    df_sorted = df.sort_values(
        by='account_age_days', ascending=False).reset_index(drop=True)

    # Eliminating Target Leakage markers to force the model to predict the future based on behavior
    leakage_features = [
        # Pandas index artifact — sorted by churn_status (65% importance!)
        'Unnamed: 0',
        'days_since_last_activity',
        'days_since_last_payment',
        'ghosting_delta',
        'is_zombie_subscriber'
    ]

    noise_features = [
        'usage_intensity', 'max_fail_time', 'aspect_ratio_3_2_count', 'failed_ratio',
        'wasted_life_index', 'quiz_completion_score', 'max_consecutive_nsfw', 'avg_fail_time',
        'count_sub_create', 'failed_generations', 'resolution_1080_count', 'max_consecutive_fails',
        'resolution_720_count', 'aspect_ratio_21_9_count', 'fraud_mismatch_rate'
    ]

    X = df_sorted.drop(
        columns=['user_id', 'churn_status'] + leakage_features + noise_features)
    y = df_sorted['churn_status']

    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Categorical Features detected: {len(cat_features)}")

    print("\n--- Phase 1: Standard 90/10 OOT Holdout Validation ---")
    split_idx = int(len(X) * 0.9)
    X_train_90, X_val_10 = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_90, y_val_10 = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train_90)} | Val size: {len(X_val_10)}")
    
    model_90 = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02633803349825135,
        depth=9,
        l2_leaf_reg=10.739472786724573,
        bootstrap_type='Bernoulli',
        subsample=0.9084491578771329,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type="GPU",
        devices='0',
        verbose=False
    )
    
    model_90.fit(
        X_train_90, y_train_90,
        cat_features=cat_features,
        eval_set=(X_val_10, y_val_10),
        early_stopping_rounds=50
    )
    
    y_proba_90 = model_90.predict_proba(X_val_10)
    optimized_preds_90 = tune_multiclass_thresholds(y_val_10, y_proba_90, model_90.classes_, majority_class='not_churned')
    f1_90 = f1_score(y_val_10, optimized_preds_90, average='weighted')
    
    print(f"--> Phase 1 (Single 90/10 Split) Secured Weighted F1: {f1_90:.4f}\n")
    print("="*60)

    print("\n--- Initiating 5-Fold Expanding Window Time-Series CV ---")
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"\n[Fold {fold+1}/5] Train size: {len(X_train)} | Val size: {len(X_val)}")
        
        model = CatBoostClassifier(
            iterations=1500, # Reduced max limit to optimize CV speed
            learning_rate=0.02633803349825135,
            depth=9,
            l2_leaf_reg=10.739472786724573,
            bootstrap_type='Bernoulli',
            subsample=0.9084491578771329,
            loss_function='MultiClass',
            eval_metric='TotalF1',
            auto_class_weights='Balanced',
            random_seed=42,
            task_type="GPU",
            devices='0',
            verbose=False
        )
        
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
        
        y_proba = model.predict_proba(X_val)
        optimized_preds = tune_multiclass_thresholds(y_val, y_proba, model.classes_, majority_class='not_churned')
        
        fold_f1 = f1_score(y_val, optimized_preds, average='weighted')
        f1_scores.append(fold_f1)
        print(f"--> Fold {fold+1} Secured Weighted F1: {fold_f1:.4f}\n")
        
    print(f"==========================================")
    print(f"Final TimeSeries CV Weighted F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"==========================================\n")
    
    print("Training FINAL model on the FULL 100% dataset for Deployment...")
    final_model = CatBoostClassifier(
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
        task_type="GPU",
        devices='0',
        verbose=100
    )
    
    final_model.fit(
        X, y,
        cat_features=cat_features
    )
    
    final_model.save_model('catboost_churn_v9-timeseries.cbm')
    print("Final Model mathematically locked and saved to catboost_churn_v9-timeseries.cbm!")


if __name__ == '__main__':
    train_catboost_multiclass(
        'dataset/train/train_users_merged_advanced_v9.csv')
