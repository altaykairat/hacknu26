import csv
from collections import defaultdict
from datetime import datetime, timedelta
import sys

def parse_date(date_str):
    if not date_str:
        return None
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def extract_mode(counts_dict, default_val="Auto"):
    if not counts_dict:
        return default_val
    best_k = max(counts_dict.items(), key=lambda item: item[1])[0]
    return best_k if best_k else default_val

def process_advanced_metrics(users_file, generations_file, output_file):
    print(f"\n[{datetime.now()}] Initializing Advanced V2 Aggregation...")
    
    # 1. Load the original users into a dict mapping uid -> data_dict
    user_data = {}
    users_headers = []
    
    with open(users_file, mode='r', encoding='utf-8') as uf:
        reader = csv.DictReader(uf)
        users_headers = list(reader.fieldnames)
        
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            user_data[uid] = dict(row)
            
            # Setup mathematical defaults
            user_data[uid].update({
                'success_time_sum': 0.0,
                'success_time_count': 0,
                'fail_time_sum': 0.0,
                'fail_time_count': 0,
                
                'total_generations': 0,
                'completed': 0,
                'failed': 0,
                'nsfw': 0,
                
                'unique_active_days': set(),
                'daily_gen_counts': defaultdict(int),
                'max_created_at': None,
                
                'total_credit_spent': 0.0,
                'duration_sum': 0.0,
                'free_generations_count': 0,
                
                'aspect_ratio_counts': defaultdict(int),
                'resolution_counts': defaultdict(int),
                'generation_type_counts': defaultdict(int),
                'duration_counts': defaultdict(int),
                
                'max_credit_cost': 0.0,
                'max_success_time': 0.0,
                'max_fail_time': 0.0,
                
                'current_fail_streak': 0,
                'max_fail_streak': 0,
                'current_nsfw_streak': 0,
                'max_nsfw_streak': 0
            })
            
    print(f"[{datetime.now()}] Pre-loaded {len(user_data)} users.")
    
    global_max_date = None
    
    # 2. Iterate the massive generations file
    print(f"[{datetime.now()}] Streaming {generations_file} ...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        headers = next(reader)
        
        uid_idx = headers.index('user_id')
        created_idx = headers.index('created_at')
        completed_idx = headers.index('completed_at')
        failed_idx = headers.index('failed_at')
        status_idx = headers.index('status')
        cost_idx = headers.index('credit_cost')
        dur_idx = headers.index('duration')
        
        type_idx = headers.index('generation_type')
        res_idx = headers.index('resolution')
        # Handle typo in header 'aspect_ration'
        try:
            ar_idx = headers.index('aspect_ration')
        except:
            ar_idx = headers.index('aspect_ratio')
            
        for row in reader:
            if len(row) <= max(uid_idx, dur_idx, res_idx, ar_idx):
                continue
                
            uid = row[uid_idx].strip()
            if uid not in user_data:
                continue
                
            ud = user_data[uid]
            status = row[status_idx].strip().lower()
            
            created_dt = parse_date(row[created_idx].strip())
            
            ud['total_generations'] += 1
            
            # --- Categorical string metrics ---
            res_str = row[res_idx].strip() if row[res_idx].strip() else "Auto"
            ar_str = row[ar_idx].strip() if row[ar_idx].strip() else "Auto"
            dur_str = row[dur_idx].strip() if row[dur_idx].strip() else "Auto"
            gen_type_str = row[type_idx].strip() if row[type_idx].strip() else "Auto"
            
            ud['resolution_counts'][res_str] += 1
            ud['aspect_ratio_counts'][ar_str] += 1
            ud['duration_counts'][dur_str] += 1
            ud['generation_type_counts'][gen_type_str] += 1
            
            # --- Streaks and Status counts ---
            if status == 'completed':
                ud['completed'] += 1
                ud['current_fail_streak'] = 0
                ud['current_nsfw_streak'] = 0
            elif status == 'failed':
                ud['failed'] += 1
                ud['current_nsfw_streak'] = 0
                ud['current_fail_streak'] += 1
                if ud['current_fail_streak'] > ud['max_fail_streak']:
                    ud['max_fail_streak'] = ud['current_fail_streak']
            elif status == 'nsfw':
                ud['nsfw'] += 1
                ud['current_fail_streak'] = 0
                ud['current_nsfw_streak'] += 1
                if ud['current_nsfw_streak'] > ud['max_nsfw_streak']:
                    ud['max_nsfw_streak'] = ud['current_nsfw_streak']
            else:
                ud['current_fail_streak'] = 0
                ud['current_nsfw_streak'] = 0
                
            # --- Date tracking for Behavior Metrics ---
            if created_dt:
                date_str = created_dt.strftime('%Y-%m-%d')
                ud['unique_active_days'].add(date_str)
                ud['daily_gen_counts'][date_str] += 1
                
                if (global_max_date is None) or (created_dt > global_max_date):
                    global_max_date = created_dt
                if (ud['max_created_at'] is None) or (created_dt > ud['max_created_at']):
                    ud['max_created_at'] = created_dt
                    
            # --- Temporal metrics logic block ---
            if status == 'completed':
                comp_dt = parse_date(row[completed_idx].strip())
                if created_dt and comp_dt:
                    diff_sec = (comp_dt - created_dt).total_seconds()
                    ud['success_time_sum'] += diff_sec
                    ud['success_time_count'] += 1
                    if diff_sec > ud['max_success_time']:
                        ud['max_success_time'] = diff_sec
            elif status in ('failed', 'nsfw'):
                fail_dt = parse_date(row[failed_idx].strip())
                if created_dt and fail_dt:
                    diff_sec = (fail_dt - created_dt).total_seconds()
                    ud['fail_time_sum'] += diff_sec
                    ud['fail_time_count'] += 1
                    if diff_sec > ud['max_fail_time']:
                        ud['max_fail_time'] = diff_sec
                        
            # --- Financial metrics ---
            cost_str = row[cost_idx].strip()
            if cost_str:
                try:
                    c = float(cost_str)
                    if c < 0.000001:
                        c = 0.0
                        ud['free_generations_count'] += 1
                    ud['total_credit_spent'] += c
                    if c > ud['max_credit_cost']:
                        ud['max_credit_cost'] = c
                except ValueError: pass
                
            if dur_str and dur_str != "Auto":
                try: ud['duration_sum'] += float(dur_str)
                except ValueError: pass
                    
            count += 1
            if count % 2000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")

    print(f"[{datetime.now()}] Target reached! {count} rows parsed.")

    # 3. Output Headers Construction
    calculated_headers = [
        'total_generations',
        'completed_generations',
        'failed_generations',
        'nsfw_generations',
        
        'days_since_last_activity', 
        'unique_active_days',
        'activity_drop_ratio',
        
        'avg_success_time', 
        'max_success_time',
        'avg_fail_time',
        'max_fail_time',
        'failed_ratio', 
        'nsfw_ratio',
        
        'max_consecutive_fails',
        'max_consecutive_nsfw',
        
        'total_credit_spent', 
        'avg_credit_per_gen',
        'max_credit_cost',
        'free_generations_count',
        
        'avg_duration_requested',
        
        'mode_aspect_ratio',
        'mode_resolution',
        'mode_generation_type',
        'mode_duration',
        
        'user_count_720p',
        'user_count_1080p',
        'user_gen_count_resolution_auto',
        'user_gen_count_aspect_ratio_auto',
        'user_gen_count_duration_auto'
    ]
    
    out_headers = users_headers + calculated_headers
    
    print(f"[{datetime.now()}] Calculating Laplace Smoothing Logic and writing output...")
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for uid, ud in user_data.items():
            tot = ud['total_generations']
            if tot == 0:
                final_row = {k: ud.get(k, 0.0) for k in out_headers}
                # override textual defaults
                for col in ['mode_aspect_ratio', 'mode_resolution', 'mode_generation_type', 'mode_duration']:
                    final_row[col] = "Auto"
                writer.writerow(final_row)
                continue
                
            # Arithmetics
            avg_succ = ud['success_time_sum'] / ud['success_time_count'] if ud['success_time_count'] > 0 else 0.0
            avg_fail = ud['fail_time_sum'] / ud['fail_time_count'] if ud['fail_time_count'] > 0 else 0.0
            f_ratio = ud['failed'] / tot
            n_ratio = ud['nsfw'] / tot
            days_since = (global_max_date - ud['max_created_at']).days if (global_max_date and ud['max_created_at']) else 0
            
            # --- Activity Drop Ratio (Laplace Smoothing) ---
            last_7_counts = 0
            prev_7_counts = 0
            if ud['max_created_at']:
                m_date = ud['max_created_at'].date()
                for d_str, dt_count in ud['daily_gen_counts'].items():
                    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
                    delta = (m_date - d_obj).days
                    if 0 <= delta <= 6:
                        last_7_counts += dt_count
                    elif 7 <= delta <= 13:
                        prev_7_counts += dt_count
                        
            # (Last 7 + 1) / (Prev 7 + 1)
            activity_drop_ratio = (last_7_counts + 1) / (prev_7_counts + 1)
            
            # Mapping definitions safely back to dictionary
            ud['completed_generations'] = ud['completed']
            ud['failed_generations'] = ud['failed']
            ud['nsfw_generations'] = ud['nsfw']
            
            ud['avg_success_time'] = round(avg_succ, 3)
            ud['avg_fail_time'] = round(avg_fail, 3)
            ud['failed_ratio'] = round(f_ratio, 4)
            ud['nsfw_ratio'] = round(n_ratio, 4)
            
            ud['days_since_last_activity'] = max(0, days_since)
            ud['unique_active_days'] = len(ud['unique_active_days'])
            ud['activity_drop_ratio'] = round(activity_drop_ratio, 4)
            
            ud['max_consecutive_fails'] = ud['max_fail_streak']
            ud['max_consecutive_nsfw'] = ud['max_nsfw_streak']
            
            ud['total_credit_spent'] = round(ud['total_credit_spent'], 2)
            ud['avg_credit_per_gen'] = round(ud['total_credit_spent'] / tot, 2)
            ud['max_credit_cost'] = round(ud['max_credit_cost'], 2)
            ud['avg_duration_requested'] = round(ud['duration_sum'] / tot, 2)
            
            # Extract Modes
            ud['mode_aspect_ratio'] = extract_mode(ud['aspect_ratio_counts'])
            ud['mode_resolution'] = extract_mode(ud['resolution_counts'])
            ud['mode_generation_type'] = extract_mode(ud['generation_type_counts'])
            ud['mode_duration'] = extract_mode(ud['duration_counts'])
            
            # Explicit Individual Constraints
            ud['user_count_720p'] = ud['resolution_counts'].get('720p', 0)
            ud['user_count_1080p'] = ud['resolution_counts'].get('1080p', 0)
            ud['user_gen_count_resolution_auto'] = ud['resolution_counts'].get('Auto', 0)
            ud['user_gen_count_aspect_ratio_auto'] = ud['aspect_ratio_counts'].get('Auto', 0)
            ud['user_gen_count_duration_auto'] = ud['duration_counts'].get('Auto', 0)
            
            final_row = {k: ud.get(k, 0) for k in out_headers}
            writer.writerow(final_row)

    print(f"[{datetime.now()}] V2 Output completed: {output_file} \n")

if __name__ == '__main__':
    print("BEGINNING TRAINING SET MERGE")
    process_advanced_metrics(
        users_file='dataset/train/train_users.csv',
        generations_file='dataset/train/train_users_generations.csv', 
        output_file='dataset/train/train_users_merged_advanced_v2.csv'
    )
    
    print("\nBEGINNING TEST SET MERGE")
    process_advanced_metrics(
        users_file='dataset/test/test_users.csv',
        generations_file='dataset/test/test_users_generations.csv', 
        output_file='dataset/test/test_users_merged_advanced_v2.csv'
    )
