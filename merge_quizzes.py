import csv
from datetime import datetime

def process_quizzes_advanced(quizzes_file):
    print(f"[{datetime.now()}] Adv-Aggregating {quizzes_file}...")
    
    quiz_keys = ['source', 'flow_type', 'team_size', 'experience', 'usage_plan', 'frustration', 'first_feature', 'role']
    users_data = {}
    
    count = 0
    with open(quizzes_file, mode='r', encoding='utf-8') as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            users_data[uid] = {}
            answered_count = 0
            
            for k in quiz_keys:
                val = row.get(k, '').strip()
                if val == '':
                    users_data[uid][k] = 'Skipped'
                else:
                    users_data[uid][k] = val
                    answered_count += 1
            
            # --- Feature 1: Quiz Completion Score ---
            users_data[uid]['quiz_completion_score'] = answered_count
            
            # --- Feature 2: B2B/B2C Segmentation ---
            flow = users_data[uid]['flow_type'].lower()
            tsize = users_data[uid]['team_size'].lower()
            
            # Identify enterprise/B2B users based on team metrics or invited flow
            if flow == 'invited' or tsize in ['small', 'midsize', 'enterprise', 'growing', '11-50', '2001-5000', '5000+', '501-2000']:
                users_data[uid]['is_B2B'] = 1
                users_data[uid]['is_B2C'] = 0
            else:
                users_data[uid]['is_B2B'] = 0
                users_data[uid]['is_B2C'] = 1
                
            # --- Feature 3: Cost Sensitivity Flag ---
            frust = users_data[uid]['frustration'].lower()
            if 'cost' in frust: # Catches 'high-cost' and 'High cost of top models' etc.
                users_data[uid]['is_cost_sensitive'] = 1
            else:
                users_data[uid]['is_cost_sensitive'] = 0
                
            count += 1
            
    print(f"[{datetime.now()}] Mapped {count} user quiz profiles with engineered metrics.")
    return users_data, quiz_keys + ['quiz_completion_score', 'is_B2B', 'is_B2C', 'is_cost_sensitive']

def merge_quizzes(v6_file, quizzes_file, output_file):
    users_data, all_keys = process_quizzes_advanced(quizzes_file)
    
    print(f"[{datetime.now()}] Left Merging quizzes into {v6_file} ...")
    
    with open(v6_file, mode='r', encoding='utf-8') as vf:
        reader = csv.DictReader(vf)
        out_headers = list(reader.fieldnames) + all_keys
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=out_headers)
            writer.writeheader()
            
            count = 0
            for row in reader:
                uid = row.get('user_id', '')
                
                u_data = users_data.get(uid)
                if u_data is None:
                    # User completely skipped onboarding (did not take quiz)
                    metrics = {k: 'Skipped' for k in all_keys if k not in ['quiz_completion_score', 'is_B2B', 'is_B2C', 'is_cost_sensitive']}
                    metrics.update({
                        'quiz_completion_score': 0,
                        'is_B2B': 0, # Default to B2C or unseen
                        'is_B2C': 1,
                        'is_cost_sensitive': 0
                    })
                else:
                    metrics = u_data
                
                row.update(metrics)
                writer.writerow(row)
                count += 1
                
    print(f"[{datetime.now()}] Attached categorical features & flags into {output_file} successfully!\n")

if __name__ == '__main__':
    print("--- MERGING ADVANCED QUIZZES FOR TRAIN SET ---")
    merge_quizzes(
        v6_file='dataset/train/train_users_merged_advanced_v6.csv',
        quizzes_file='dataset/train/train_users_quizzes.csv',
        output_file='dataset/train/train_users_merged_advanced_v7.csv'
    )
    
    print("--- MERGING ADVANCED QUIZZES FOR TEST SET ---")
    merge_quizzes(
        v6_file='dataset/test/test_users_merged_advanced_v6.csv',
        quizzes_file='dataset/test/test_users_quizzes.csv',
        output_file='dataset/test/test_users_merged_advanced_v7.csv'
    )
