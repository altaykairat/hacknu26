import csv
from datetime import datetime
from collections import defaultdict

def fix_date(date_str):
    if not date_str:
        return None
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def process_purchases(purchases_file):
    print(f"[{datetime.now()}] Reading and aggregating {purchases_file}...")
    
    global_max_date = None
    
    # Store aggregated metrics per user
    users_data = defaultdict(lambda: {
        'total_dollars_spent': 0.0,
        'max_purchase_amount': 0.0,
        'total_transactions': 0,
        'last_payment_date': None,
        'count_credits_package': 0,
        'count_sub_create': 0,
        'count_sub_update': 0
    })
    
    count = 0
    with open(purchases_file, mode='r', encoding='utf-8') as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            count += 1
            
            # Parse Date
            dt_str = row.get('purchase_time', '')
            dt = fix_date(dt_str)
            if dt:
                if global_max_date is None or dt > global_max_date:
                    global_max_date = dt
                
                current_user_max = users_data[uid]['last_payment_date']
                if current_user_max is None or dt > current_user_max:
                    users_data[uid]['last_payment_date'] = dt
                    
            # Parse Amount
            amt_str = row.get('purchase_amount_dollars', '0.0')
            try:
                amt = float(amt_str)
            except ValueError:
                amt = 0.0
                
            users_data[uid]['total_dollars_spent'] += amt
            if amt > users_data[uid]['max_purchase_amount']:
                users_data[uid]['max_purchase_amount'] = amt
            users_data[uid]['total_transactions'] += 1
            
            # Parse Types
            ptype = row.get('purchase_type', '').lower()
            if 'credits package' in ptype:
                users_data[uid]['count_credits_package'] += 1
            elif 'create' in ptype:
                users_data[uid]['count_sub_create'] += 1
            elif 'update' in ptype:
                users_data[uid]['count_sub_update'] += 1
                
    print(f"[{datetime.now()}] Parsed {count} transactions across {len(users_data)} users.")
    print(f"[{datetime.now()}] Global Max Payment Date: {global_max_date}")
    return users_data, global_max_date

def merge_purchases(v5_file, purchases_file, output_file):
    users_data, global_max_date = process_purchases(purchases_file)
    
    print(f"[{datetime.now()}] Streaming and updating {v5_file} with financial features...")
    
    with open(v5_file, mode='r', encoding='utf-8') as vf:
        reader = csv.DictReader(vf)
        
        new_cols = [
            'total_dollars_spent', 
            'max_purchase_amount', 
            'avg_purchase_amount', 
            'total_transactions', 
            'days_since_last_payment',
            'count_credits_package',
            'count_sub_create',
            'count_sub_update'
        ]
        out_headers = list(reader.fieldnames) + new_cols
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=out_headers)
            writer.writeheader()
            
            count = 0
            for row in reader:
                uid = row.get('user_id', '')
                
                # Extract aggregated logic
                u_data = users_data.get(uid)
                
                if u_data is None:
                    # User made 0 monetary purchases
                    metrics = {
                        'total_dollars_spent': 0.0,
                        'max_purchase_amount': 0.0,
                        'avg_purchase_amount': 0.0,
                        'total_transactions': 0,
                        'days_since_last_payment': 999.0, # Safe arbitrary high constant for tree-split
                        'count_credits_package': 0,
                        'count_sub_create': 0,
                        'count_sub_update': 0
                    }
                else:
                    avg_amt = 0.0
                    if u_data['total_transactions'] > 0:
                        avg_amt = u_data['total_dollars_spent'] / u_data['total_transactions']
                        
                    days_since = 999.0
                    if u_data['last_payment_date'] and global_max_date:
                        days_since = max(0, (global_max_date - u_data['last_payment_date']).days)
                        
                    metrics = {
                        'total_dollars_spent': round(u_data['total_dollars_spent'], 2),
                        'max_purchase_amount': round(u_data['max_purchase_amount'], 2),
                        'avg_purchase_amount': round(avg_amt, 2),
                        'total_transactions': u_data['total_transactions'],
                        'days_since_last_payment': days_since,
                        'count_credits_package': u_data['count_credits_package'],
                        'count_sub_create': u_data['count_sub_create'],
                        'count_sub_update': u_data['count_sub_update']
                    }
                
                row.update(metrics)
                writer.writerow(row)
                count += 1
                
    print(f"[{datetime.now()}] Seamlessly merged financial records into {output_file}!\n")

if __name__ == '__main__':
    print("--- COMPUTING FINANCIAL METRICS FOR TRAIN SET ---")
    merge_purchases(
        v5_file='dataset/train/train_users_merged_advanced_v5.csv',
        purchases_file='dataset/train/train_users_purchases.csv',
        output_file='dataset/train/train_users_merged_advanced_v6.csv'
    )
    
    print("--- COMPUTING FINANCIAL METRICS FOR TEST SET ---")
    merge_purchases(
        v5_file='dataset/test/test_users_merged_advanced_v5.csv',
        purchases_file='dataset/test/test_users_purchases.csv',
        output_file='dataset/test/test_users_merged_advanced_v6.csv'
    )
    
