import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# 1. Plotting the Expanding Window CV Trend
def plot_cv_trend():
    plt.style.use('dark_background') # Better for dark presentations
    folds = ['Fold 1\n(15k)', 'Fold 2\n(30k)', 'Fold 3\n(45k)', 'Fold 4\n(60k)', 'Fold 5\n(75k)']
    f1_scores = [0.5541, 0.3750, 0.5229, 0.5013, 0.5060] # Representing the true trend we saw in logs
    
    plt.figure(figsize=(12, 6))
    plt.plot(folds, f1_scores, marker='o', markersize=12, linestyle='-', linewidth=4, color='#00ffcc')
    plt.axhline(y=0.50, color='#ff0055', linestyle='--', linewidth=2, label='Baseline Target (0.50 F1)')
    
    plt.title('Mathematical Stability: Expanding Window Time-Series CV', fontsize=18, pad=20, color='white')
    plt.ylabel('Weighted F1 Score', fontsize=14, color='white')
    plt.xlabel('Chronological Data Intake (Users in Training Matrix)', fontsize=14, color='white', labelpad=15)
    plt.ylim(0.30, 0.65)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.legend(fontsize=12, facecolor='black', edgecolor='white')
    plt.tight_layout()
    plt.savefig('presentation_cv_trend.png', dpi=300, transparent=True)
    print("Saved: presentation_cv_trend.png")

# 2. Extracting Top 15 Feature Importances from the exact CBM matrix
def plot_feature_importances(model_path):
    plt.style.use('dark_background')
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    importances = model.get_feature_importance()
    features = model.feature_names_
    
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
    plt.title('Top 15 Absolute Churn Drivers (v9 Timeseries Matrix)', fontsize=18, color='white')
    plt.xlabel('Feature Importance (%)', fontsize=14, color='white')
    plt.ylabel('')
    plt.xticks(color='white')
    plt.yticks(color='white', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    plt.savefig('presentation_feature_importance.png', dpi=300, transparent=True)
    print("Saved: presentation_feature_importance.png")

# 3. SHAP Summary Plot
def generate_shap(model_path, data_path):
    print("Loading data for SHAP analysis (using 3000 sampled users to respect RAM)...")
    plt.style.use('default') # SHAP plots look best on default white background due to color scaling
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    df = pd.read_csv(data_path)
    df.fillna('Unknown', inplace=True)
    
    leakage_features = [
        'Unnamed: 0', 'days_since_last_activity', 'days_since_last_payment',
        'ghosting_delta', 'is_zombie_subscriber'
    ]
    noise_features = [
        'usage_intensity', 'max_fail_time', 'aspect_ratio_3_2_count', 'failed_ratio',
        'wasted_life_index', 'quiz_completion_score', 'max_consecutive_nsfw', 'avg_fail_time',
        'count_sub_create', 'failed_generations', 'resolution_1080_count', 'max_consecutive_fails',
        'resolution_720_count', 'aspect_ratio_21_9_count', 'fraud_mismatch_rate'
    ]
    
    drop_cols = [col for col in (leakage_features + noise_features) if col in df.columns]
    X = df.drop(columns=['user_id', 'churn_status'] + drop_cols)
    
    X_sample = X.sample(n=3000, random_state=42)
    
    print("Computing SHAP matrices...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Iterate through classes and generate separate graphs for purely Churn variables
    for idx, cls in enumerate(model.classes_):
        if cls == 'not_churned':
            continue 
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[idx], X_sample, max_display=15, show=False)
        plt.title(f'SHAP Velocity Drivers for [{cls}]', fontsize=18, y=1.05)
        plt.tight_layout()
        plt.savefig(f'presentation_shap_{cls}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: presentation_shap_{cls}.png")


if __name__ == "__main__":
    plot_cv_trend()
    plot_feature_importances('catboost_churn.cbm')
    generate_shap('catboost_churn.cbm', 'dataset/train/train_users_merged_advanced_v9.csv')
