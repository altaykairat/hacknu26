import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, f1_score


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

        # Calculate Macro F1 (crucial for imbalanced multi-class)
        current_f1 = f1_score(y_true, y_pred_custom, average='macro')

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_preds = y_pred_custom
            best_thresh = thresh

    print(
        f"--> Optimal Churn Threshold Found: {best_thresh:.2f} (Macro F1: {best_f1:.4f})")
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
    X = df_sorted.drop(columns=['user_id', 'churn_status'] + leakage_features)
    y = df_sorted['churn_status']

    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    split_idx = int(len(X) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Categorical Features detected: {len(cat_features)}")

    # 3. Class Weights (Manual Control over the 2:1:1 split)
    # Target values: 'not_churned' (50%), 'invol_churn' (25%), 'vol_churn' (25%)
    # Let's manually up-weight the churn vectors slightly beyond strict math to prioritize recall
    classes = sorted(y_train.unique())
    # E.g. ['invol_churn', 'not_churned', 'vol_churn'] -> we can map them dynamically
    class_weights_dict = {
        'not_churned': 1.0,
        'invol_churn': 2.5,  # Mathematically 2.0, boosted to 2.5 for tuning
        'vol_churn': 2.5     # Mathematically 2.0, boosted to 2.5 for tuning
    }

    model = CatBoostClassifier(
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
        devices='0'            
    )

    print("Beginning Training phase...")
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100
    )

    # 4. Evaluation
    preds = model.predict(X_val)
    print("\n--- OOT Validation Classification Report (Default Threshold) ---")
    print(classification_report(y_val, preds))

    # --- Threshold Tuning Injection ---
    print("\nApplying Custom Threshold Tuning...")
    y_proba = model.predict_proba(X_val)
    optimized_preds = tune_multiclass_thresholds(
        y_val, y_proba, model.classes_, majority_class='not_churned')

    print("\n--- OOT Validation Classification Report (Optimized Thresholds) ---")
    print(classification_report(y_val, optimized_preds))

    model.save_model('catboost_churn_v9.cbm')
    print("Model saved to catboost_churn_v9.cbm!")


if __name__ == '__main__':
    train_catboost_multiclass(
        'dataset/train/train_users_merged_advanced_v9.csv')
