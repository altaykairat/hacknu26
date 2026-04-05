import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# Quiet version of our tuning function
def tune_multiclass_thresholds_quiet(y_true, y_proba, classes, majority_class='not_churned'):
    best_f1 = 0
    best_preds = None
    
    try:
        majority_idx = list(classes).index(majority_class)
    except ValueError:
        majority_idx = 0
        
    for thresh in np.arange(0.20, 0.55, 0.02):
        y_pred_custom = []
        for probas in y_proba:
            churn_probs = [p for i, p in enumerate(probas) if i != majority_idx]
            max_churn_prob = max(churn_probs)
            max_churn_idx = list(probas).index(max_churn_prob)
            
            if max_churn_prob >= thresh:
                y_pred_custom.append(classes[max_churn_idx])
            else:
                y_pred_custom.append(classes[majority_idx])
                
        current_f1 = f1_score(y_true, y_pred_custom, average='macro')
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_preds = y_pred_custom
            
    return best_f1

def objective(trial):
    # Data loading exactly replicating our true behavioral framework
    df = pd.read_csv('dataset/train/train_users_merged_advanced_v9.csv')
    df.fillna('Unknown', inplace=True)
    df_sorted = df.sort_values(by='account_age_days', ascending=False).reset_index(drop=True)
    
    leakage_features = [
        'Unnamed: 0', 'days_since_last_activity', 
        'days_since_last_payment', 'ghosting_delta', 'is_zombie_subscriber'
    ]
    X = df_sorted.drop(columns=['user_id', 'churn_status'] + leakage_features)
    y = df_sorted['churn_status']
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # ---------------------------------------------------------
    # Building the Search Space
    # ---------------------------------------------------------
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
        
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',        # Assuming hardware is the RTX 3050 (0)
        'verbose': False
    }

    # Bernoulli allows stochastic subsampling dropping rows
    if params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 0.95)
        
    # Fit strictly with Early Stopping to save sweep compute
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Calculate Macro F1 using our specific Threshold logic map
    y_proba = model.predict_proba(X_val)
    macro_f1 = tune_multiclass_thresholds_quiet(y_val, y_proba, model.classes_)
    
    return macro_f1

if __name__ == '__main__':
    print("Initiating Optuna Hyperparameter Study (20 Trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # TPESampler handles Bayes optimization natively
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Macro F1: {study.best_value:.4f}")
    print("Best Params:")
    for key, val in study.best_params.items():
        print(f"    {key}: {val}")
