import pandas as pd
from catboost import CatBoostClassifier

def generate_submission(model_path, test_csv_path, output_csv='submission.csv'):
    print(f"Loading CatBoost model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    print(f"Loading test dataset from {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    df_test.fillna('Unknown', inplace=True)
    
    # Ensure exact same feature structures are dropped
    leakage_features = [
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
    
    # Intersect to safely drop without KeyError
    drop_cols = [col for col in (leakage_features + noise_features) if col in df_test.columns]
    
    print("Extracting features (applying categorical formatting)...")
    X_test = df_test.drop(columns=['user_id'] + drop_cols)
    if 'churn_status' in X_test.columns:
        X_test = X_test.drop(columns=['churn_status'])
        
    print("Executing GPU Inference...")
    
    # We apply the manual 0.46 Threshold logic from our latest OOT training metric
    y_proba = model.predict_proba(X_test)
    classes = model.classes_
    majority_idx = list(classes).index('not_churned')
    thresh = 0.46
    
    y_pred_custom = []
    for probas in y_proba:
        churn_probs = [p for i, p in enumerate(probas) if i != majority_idx]
        max_churn_prob = max(churn_probs)
        max_churn_idx = list(probas).index(max_churn_prob)

        if max_churn_prob >= thresh:
            y_pred_custom.append(classes[max_churn_idx])
        else:
            y_pred_custom.append(classes[majority_idx])
    
    print("Packaging predictions into final format...")
    submission = pd.DataFrame({
        'user_id': df_test['user_id'],
        'churn_status': y_pred_custom
    })
    
    submission.to_csv(output_csv, index=False)
    print(f"SUCCESS: Submission saved to {output_csv}!")

if __name__ == '__main__':
    generate_submission(
        model_path='catboost_churn_v9-timeseries.cbm', # Using the TS pruned matrix
        test_csv_path='dataset/test/test_users_merged_advanced_v9.csv',
        output_csv='submission.csv'
    )
