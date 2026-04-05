import csv
from datetime import datetime

def generate_feature_crosses(v8_file, output_file):
    print(f"[{datetime.now()}] Reading master dataset {v8_file} for psychological feature crosses...")
    
    new_keys = [
        'ghosting_delta',
        'real_cost_per_generation',
        'usage_intensity',
        'freeloader_ratio',
        'total_payment_friction',
        'wasted_life_index',
        'is_zombie_subscriber',
        'explorer_score',
        'overwhelmed_beginner_flag',
        'dollars_per_active_day'
    ]
    
    with open(v8_file, mode='r', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        out_headers = list(reader.fieldnames) + new_keys
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=out_headers)
            writer.writeheader()
            
            count = 0
            for row in reader:
                # Safely parse numeric base fields
                try:
                    days_since_last_payment = float(row.get('days_since_last_payment', 0))
                except ValueError:
                    days_since_last_payment = 0.0
                    
                days_since_last_activity = float(row.get('days_since_last_activity', 0) or 0)
                total_dollars_spent = float(row.get('total_dollars_spent', 0) or 0)
                total_generations = float(row.get('total_generations', 0) or 0)
                account_age_days = float(row.get('account_age_days', 0) or 0)
                free_generations_count = float(row.get('free_generations_count', 0) or 0)
                
                payment_failure_rate = float(row.get('payment_failure_rate', 0) or 0)
                fraud_mismatch_rate = float(row.get('fraud_mismatch_rate', 0) or 0)
                unauth_3ds = float(row.get('unauthenticated_3ds_count', 0) or 0)
                total_tx = float(row.get('total_transactions', 0) or 0)

                # New Psych Variables variables
                def safe_float(val):
                    if val == 'set()': return 0.0
                    try:
                        return float(val) if val else 0.0
                    except (ValueError, TypeError):
                        return 0.0

                failed_gen = safe_float(row.get('failed_generations'))
                avg_fail_time = safe_float(row.get('avg_fail_time'))
                unique_active_days = safe_float(row.get('unique_active_days'))
                experience = row.get('experience', '').strip().lower()
                mode_resolution = row.get('mode_resolution', '').strip().lower()
                avg_credit_per_gen = safe_float(row.get('avg_credit_per_gen'))

                # --- 1. Original 5 Crosses ---
                ghosting_delta = days_since_last_payment - days_since_last_activity
                real_cost_per_gen = round(total_dollars_spent / total_generations, 4) if total_generations > 0 else 0.0
                usage_intensity = round(total_generations / account_age_days, 4) if account_age_days > 0 else 0.0
                freeloader_ratio = round(free_generations_count / total_generations, 4) if total_generations > 0 else 0.0
                friction_ratio = unauth_3ds / total_tx if total_tx > 0 else 0.0
                total_payment_friction = round(payment_failure_rate + fraud_mismatch_rate + friction_ratio, 4)

                # --- 2. Advanced Psychological Crosses ---
                
                # 2.1 Wasted Life Index (total seconds wasted on failed generations)
                wasted_life_index = round(failed_gen * avg_fail_time, 2)
                
                # 2.2 Zombie Subscriber Flag (paid within last 30 days, but hasn't logged in for 3+ weeks)
                is_zombie_subscriber = 1 if (days_since_last_payment < 30 and days_since_last_activity > 21) else 0
                
                # 2.3 Explorer Score (counts how many distinct valid resolution buckets they used)
                explorer_score = 0
                for col, val in row.items():
                    if col and col.startswith('resolution_') and col.endswith('_count'):
                        v = float(val) if val else 0
                        if v > 0:
                            explorer_score += 1
                            
                # 2.4 Overwhelmed Beginner Flag (beginner using 4K or burning 300+ credits on average)
                overwhelmed_beginner_flag = 1 if (experience == 'beginner' and (mode_resolution == '4k' or avg_credit_per_gen > 300)) else 0
                
                # 2.5 Dollars per Active Day (Financial explicit expectation)
                dollars_per_active_day = round(total_dollars_spent / unique_active_days, 4) if unique_active_days > 0 else 0.0

                # Attach all 10 composite metrics
                row.update({
                    'ghosting_delta': round(ghosting_delta, 2),
                    'real_cost_per_generation': real_cost_per_gen,
                    'usage_intensity': usage_intensity,
                    'freeloader_ratio': freeloader_ratio,
                    'total_payment_friction': total_payment_friction,
                    'wasted_life_index': wasted_life_index,
                    'is_zombie_subscriber': is_zombie_subscriber,
                    'explorer_score': explorer_score,
                    'overwhelmed_beginner_flag': overwhelmed_beginner_flag,
                    'dollars_per_active_day': dollars_per_active_day
                })
                
                writer.writerow(row)
                count += 1
                
    print(f"[{datetime.now()}] Engineered 10 psychological & composite crosses for {count} users into {output_file} successfully!\n")

if __name__ == '__main__':
    print("--- COMPUTING 10 FEATURE CROSSES FOR TRAIN SET ---")
    generate_feature_crosses(
        v8_file='dataset/train/train_users_merged_advanced_v8.csv',
        output_file='dataset/train/train_users_merged_advanced_v9.csv'
    )
    
    print("--- COMPUTING 10 FEATURE CROSSES FOR TEST SET ---")
    generate_feature_crosses(
        v8_file='dataset/test/test_users_merged_advanced_v8.csv',
        output_file='dataset/test/test_users_merged_advanced_v9.csv'
    )
