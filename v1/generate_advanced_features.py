import csv
from collections import defaultdict
from datetime import datetime
import sys

def parse_date(date_str):
    if not date_str:
        return None
    # Fix the 1067 year anomaly by forcefully replacing it with 2023
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        # fromisoformat perfectly handles "2023-11-23 18:48:48.429383+00:00"
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def process_advanced_metrics(users_file, generations_file, output_file):
    print(f"\n[{datetime.now()}] Initializing Advanced Aggregation...")
    
    # 1. Load the original users into a dict mapping uid -> data_dict
    user_data = {}
    users_headers = []
    
    print(f"[{datetime.now()}] Pre-loading Base Users list from {users_file}...")
    with open(users_file, mode='r', encoding='utf-8') as uf:
        reader = csv.DictReader(uf)
        users_headers = list(reader.fieldnames)
        
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            # Keep original data
            user_data[uid] = dict(row)
            
            # Setup mathematical defaults
            user_data[uid].update({
                'success_time_sum': 0.0,
                'success_time_count': 0,
                'fail_time_sum': 0.0,
                'fail_time_count': 0,
                'total_generations': 0,
                'failed': 0,
                'nsfw': 0,
                'unique_active_days': set(),
                'max_created_at': None,
                'total_credit_spent': 0.0,
                'duration_sum': 0.0,
                'free_generations_count': 0
            })
            
    print(f"[{datetime.now()}] Pre-loaded {len(user_data)} users.")
    
    global_max_date = None
    
    # 2. Iterate the massive generations file
    print(f"[{datetime.now()}] Streaming {generations_file} ...")
    count = 0
    
    # We use csv.reader to be memory efficient and incredibly fast
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        headers = next(reader)
        
        try:
            uid_idx = headers.index('user_id')
            created_idx = headers.index('created_at')
            completed_idx = headers.index('completed_at')
            failed_idx = headers.index('failed_at')
            status_idx = headers.index('status')
            cost_idx = headers.index('credit_cost')
            dur_idx = headers.index('duration')
        except ValueError as e:
            print("Error parsing headers:", e)
            return

        for row in reader:
            if len(row) <= max(uid_idx, dur_idx):
                continue
                
            uid = row[uid_idx].strip()
            
            # Only track if UID is inside our predefined array
            if uid in user_data:
                ud = user_data[uid]
                
                status = row[status_idx].strip().lower()
                cost_str = row[cost_idx].strip()
                dur_str = row[dur_idx].strip()
                
                created_dt = parse_date(row[created_idx].strip())
                
                ud['total_generations'] += 1
                if status == 'failed':
                    ud['failed'] += 1
                elif status == 'nsfw':
                    ud['nsfw'] += 1
                    
                if created_dt:
                    if (global_max_date is None) or (created_dt > global_max_date):
                        global_max_date = created_dt
                    
                    if (ud['max_created_at'] is None) or (created_dt > ud['max_created_at']):
                        ud['max_created_at'] = created_dt
                        
                    ud['unique_active_days'].add(created_dt.strftime('%Y-%m-%d'))
                    
                if status == 'completed':
                    comp_dt = parse_date(row[completed_idx].strip())
                    if created_dt and comp_dt:
                        ud['success_time_sum'] += (comp_dt - created_dt).total_seconds()
                        ud['success_time_count'] += 1
                elif status in ('failed', 'nsfw'):
                    fail_dt = parse_date(row[failed_idx].strip())
                    if created_dt and fail_dt:
                        ud['fail_time_sum'] += (fail_dt - created_dt).total_seconds()
                        ud['fail_time_count'] += 1
                        
                if cost_str:
                    try:
                        c = float(cost_str)
                        if c < 0.000001:
                            c = 0.0
                            ud['free_generations_count'] += 1
                        ud['total_credit_spent'] += c
                    except ValueError: pass
                    
                if dur_str:
                    try: ud['duration_sum'] += float(dur_str)
                    except ValueError: pass
                    
            count += 1
            if count % 2000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")

    print(f"[{datetime.now()}] Evaluated {count} generator rows!")
    print(f"[{datetime.now()}] Calculating metrics and saving...")

    # 3. Form final file
    calculated_headers = [
        'total_generations',
        'avg_success_time', 'avg_fail_time',
        'failed_ratio', 'nsfw_ratio',
        'days_since_last_activity', 'unique_active_days',
        'total_credit_spent', 'avg_credit_per_gen',
        'avg_duration_requested', 'free_generations_count'
    ]
    out_headers = users_headers + calculated_headers
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for uid, ud in user_data.items():
            tot = ud['total_generations']
            if tot == 0:
                # Still output user, just filled with 0.0 for defaults
                ud.update({k: 0.0 for k in calculated_headers})
                # Drop tracking fields that aren't in out_headers
                final_row = {k: ud.get(k, 0.0) for k in out_headers}
                writer.writerow(final_row)
                continue
                
            avg_succ = ud['success_time_sum'] / ud['success_time_count'] if ud['success_time_count'] > 0 else 0.0
            avg_fail = ud['fail_time_sum'] / ud['fail_time_count'] if ud['fail_time_count'] > 0 else 0.0
            
            f_ratio = ud['failed'] / tot
            n_ratio = ud['nsfw'] / tot
            
            days_since = (global_max_date - ud['max_created_at']).days if (global_max_date and ud['max_created_at']) else 0
            
            # Map everything exactly to output schema
            ud['avg_success_time'] = round(avg_succ, 3)
            ud['avg_fail_time'] = round(avg_fail, 3)
            ud['failed_ratio'] = round(f_ratio, 4)
            ud['nsfw_ratio'] = round(n_ratio, 4)
            ud['days_since_last_activity'] = max(0, days_since)
            ud['unique_active_days'] = len(ud['unique_active_days'])
            ud['total_credit_spent'] = round(ud['total_credit_spent'], 2)
            ud['avg_credit_per_gen'] = round(ud['total_credit_spent'] / tot, 2)
            ud['avg_duration_requested'] = round(ud['duration_sum'] / tot, 2)
            # free_generations_count and total_generations are already tracking values
            
            # Form single row dictionary safely filtered by headers
            final_row = {k: ud.get(k, 0) for k in out_headers}
            writer.writerow(final_row)

    print(f"[{datetime.now()}] Output completed: {output_file} \n")

if __name__ == '__main__':
    print("BEGINNING TRAINING SET MERGE")
    process_advanced_metrics(
        users_file='dataset/train/train_users.csv',
        generations_file='dataset/train/train_users_generations.csv', 
        output_file='dataset/train/train_users_merged_advanced.csv'
    )
    
    print("\nBEGINNING TEST SET MERGE")
    process_advanced_metrics(
        users_file='dataset/test/test_users.csv',
        generations_file='dataset/test/test_users_generations.csv', 
        output_file='dataset/test/test_users_merged_advanced.csv'
    )
