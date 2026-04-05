import csv
from datetime import datetime

def fix_date(date_str):
    if not date_str:
        return None
    # Automatically fix the 1067 anomaly year found in all dates across the dataset
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def merge_properties(v3_file, properties_file, output_file):
    print(f"\n[{datetime.now()}] Reading properties from {properties_file}...")
    
    # PASS 1: Find the Global Max Date of the dataset to calculate 'account_age_days' mathematically
    global_max_date = None
    with open(properties_file, mode='r', encoding='utf-8') as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            dt = fix_date(row.get('subscription_start_date', ''))
            if dt:
                if global_max_date is None or dt > global_max_date:
                    global_max_date = dt
                    
    print(f"[{datetime.now()}] Identified Properties Global Max Date: {global_max_date}")
    
    # PASS 2: Preload the properties table into a fast lookup dictionary (user_id -> properties)
    properties_lookup = {}
    with open(properties_file, mode='r', encoding='utf-8') as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            dt = fix_date(row.get('subscription_start_date', ''))
            account_age = 0
            if dt and global_max_date: # Calculate true age relative to the dataset max
                account_age = max(0, (global_max_date - dt).days)
            
            # Keep RAW STRING for CatBoost target-encoding natively
            plan = row.get('subscription_plan', '').strip()
            if not plan:
                plan = "Unknown"
                
            cc = row.get('country_code', '').strip()
            if not cc:
                cc = "Unknown"
                
            properties_lookup[uid] = {
                'account_age_days': account_age,
                'subscription_plan': plan,
                'country_code': cc
            }
            
    print(f"[{datetime.now()}] Pre-loaded {len(properties_lookup)} property records.")
    print(f"[{datetime.now()}] Streaming and updating base file (Left Join approach on memory) ...")
    
    with open(v3_file, mode='r', encoding='utf-8') as vf:
        reader = csv.DictReader(vf)
        
        # New columns to add mathematically appended to headers
        new_cols = ['account_age_days', 'subscription_plan', 'country_code']
        out_headers = list(reader.fieldnames) + new_cols
        
        # Stream base rows and safely write instantly to output
        with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=out_headers)
            writer.writeheader()
            
            count = 0
            for row in reader:
                uid = row.get('user_id', '')
                
                # Fetch properties or default cleanly if a user didn't have property info
                props = properties_lookup.get(uid, {
                    'account_age_days': 0,
                    'subscription_plan': 'Unknown',
                    'country_code': 'Unknown'
                })
                
                row.update(props)
                writer.writerow(row)
                count += 1
                
    print(f"[{datetime.now()}] Left Merged {count} records securely into {output_file}!")

if __name__ == '__main__':
    print("--- MERGING TRAIN SET PROPERTIES ---")
    merge_properties(
        v3_file='dataset/train/train_users_merged_advanced_v3.csv',
        properties_file='dataset/train/train_users_properties.csv',
        output_file='dataset/train/train_users_merged_advanced_v4.csv'
    )
    
    print("\n--- MERGING TEST SET PROPERTIES ---")
    merge_properties(
        v3_file='dataset/test/test_users_merged_advanced_v3.csv',
        properties_file='dataset/test/test_users_properties.csv',
        output_file='dataset/test/test_users_merged_advanced_v4.csv'
    )
