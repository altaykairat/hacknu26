import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def generate_business_insights(csv_path, model_path):
    print("Загрузка данных и модели для аналитики SHAP...")
    df = pd.read_csv(csv_path)
    df.fillna('Unknown', inplace=True)
    
    # Воспроизводим ту же логику очистки, что и при обучении
    df_sorted = df.sort_values(by='account_age_days', ascending=False).reset_index(drop=True)
    
    leakage_features = [
        'Unnamed: 0', 'days_since_last_activity', 
        'days_since_last_payment', 'ghosting_delta', 'is_zombie_subscriber'
    ]
    X = df_sorted.drop(columns=['user_id', 'churn_status'] + leakage_features)
    
    # Берем случайную выборку для ускорения расчетов SHAP 
    X_sample = X.sample(n=3000, random_state=42)
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    print("\nИнициализация SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # В MultiClass shap_values - это список массивов.
    # Индекс 0: invol_churn, Индекс 1: not_churned, Индекс 2: vol_churn (зависит от сортировки)
    classes = model.classes_
    
    for i, class_name in enumerate(classes):
        if class_name == 'not_churned':
            continue # Нас интересуют только причины оттока
            
        print(f"\nГенерация SHAP-отчета для класса: {class_name}...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, i], X_sample, show=False, max_display=15)
        
        # Настройка шрифтов и фона для темной темы презентации (опционально)
        plt.title(f"Top Churn Drivers: {class_name}", fontsize=16)
        plt.tight_layout()
        
        filename = f"shap_summary_{class_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График сохранен как {filename}")

    # Вывод в терминал для быстрого анализа
    print("\n--- Быстрый текстовый отчет (Feature Importances) ---")
    fi = model.get_feature_importance()
    fn = model.feature_names_
    top = sorted(zip(fn, fi), key=lambda x: -x[1])
    for name, imp in top[:15]:
        print(f"  {imp:6.2f}%  {name}")

if __name__ == '__main__':
    generate_business_insights(
        csv_path='dataset/train/train_users_merged_advanced_v9.csv',
        model_path='catboost_churn_v9.cbm'
    )
