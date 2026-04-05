import csv
from datetime import datetime

def fix_date(date_str):
    if not date_str:
        return None
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def process_transactions(transactions_file):
    print(f"[{datetime.now()}] Reading {transactions_file}...")
    
    users_data = {}
    
    count = 0
    with open(transactions_file, mode='r', encoding='utf-8') as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            if uid not in users_data:
                users_data[uid] = {
                    'total_attempts': 0,
                    'failed_attempts': 0,
                    'has_used_prepaid': 0,
                    'has_used_virtual': 0,
                    'fraud_mismatch_count': 0,
                    'unauthenticated_3ds_count': 0,
                    'latest_transaction_time': None,
                    'latest_cvc_check': 'unknown'
                }
                
            ud = users_data[uid]
            ud['total_attempts'] += 1
            
            # --- 1. Payment Failure Rate pre-calc ---
            if row.get('failure_code', '').strip() != '':
                ud['failed_attempts'] += 1
                
            # --- 2. Toxic Cards ---
            if row.get('is_prepaid', 'False') == 'True':
                ud['has_used_prepaid'] = 1
            if row.get('is_virtual', 'False') == 'True':
                ud['has_used_virtual'] = 1
                
            # --- 3. Fraud Mismatch ---
            billing = row.get('billing_address_country', '').strip().lower()
            card = row.get('card_country', '').strip().lower()
            bank = row.get('bank_country', '').strip().lower()
            
            # If billing exists, but differs from card OR bank
            if billing:
                if (card and billing != card) or (bank and billing != bank):
                    ud['fraud_mismatch_count'] += 1
                    
            # --- 4. 3D Secure Friction ---
            if row.get('is_3d_secure', 'False') == 'True' and row.get('is_3d_secure_authenticated', 'False') == 'False':
                ud['unauthenticated_3ds_count'] += 1
                
            # --- 5. Latest CVC Status ---
            dt = fix_date(row.get('transaction_time', ''))
            if dt:
                # If we don't have a time yet, or this time is purely newer
                if ud['latest_transaction_time'] is None or dt > ud['latest_transaction_time']:
                    ud['latest_transaction_time'] = dt
                    ud['latest_cvc_check'] = row.get('cvc_check', 'unknown').strip()
                    if ud['latest_cvc_check'] == '':
                        ud['latest_cvc_check'] = 'unknown'

            count += 1
            
    print(f"[{datetime.now()}] Mapped {count} transaction attempts.")
    return users_data

def merge_transactions(v7_file, transactions_file, output_file):
    users_data = process_transactions(transactions_file)
    
    print(f"[{datetime.now()}] Left Merging transaction metrics into {v7_file} ...")
    
    new_keys = [
        'payment_failure_rate', 
        'has_used_prepaid', 
        'has_used_virtual', 
        'fraud_mismatch_rate', 
        'unauthenticated_3ds_count',
        'latest_cvc_check'
    ]
    
    with open(v7_file, mode='r', encoding='utf-8') as vf:
        reader = csv.DictReader(vf)
        out_headers = list(reader.fieldnames) + new_keys
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=out_headers)
            writer.writeheader()
            
            count = 0
            for row in reader:
                uid = row.get('user_id', '')
                
                ud = users_data.get(uid)
                if ud is None:
                    # User never hit the billing gateway
                    metrics = {
                        'payment_failure_rate': 0.0,
                        'has_used_prepaid': 0,
                        'has_used_virtual': 0,
                        'fraud_mismatch_rate': 0.0,
                        'unauthenticated_3ds_count': 0,
                        'latest_cvc_check': 'missing'
                    }
                else:
                    tot = ud['total_attempts']
                    fail_rate = round(ud['failed_attempts'] / tot, 3) if tot > 0 else 0.0
                    fraud_rate = round(ud['fraud_mismatch_count'] / tot, 3) if tot > 0 else 0.0
                    
                    metrics = {
                        'payment_failure_rate': fail_rate,
                        'has_used_prepaid': ud['has_used_prepaid'],
                        'has_used_virtual': ud['has_used_virtual'],
                        'fraud_mismatch_rate': fraud_rate,
                        'unauthenticated_3ds_count': ud['unauthenticated_3ds_count'],
                        'latest_cvc_check': ud['latest_cvc_check']
                    }
                
                row.update(metrics)
                writer.writerow(row)
                count += 1
                
    print(f"[{datetime.now()}] Attached 6 involuntary churn gateways into {output_file} successfully!\n")

if __name__ == '__main__':
    print("--- MERGING TRANSACTIONS FOR TRAIN SET ---")
    merge_transactions(
        v7_file='dataset/train/train_users_merged_advanced_v7.csv',
        transactions_file='dataset/train/train_users_transaction_attempts_v1.csv',
        output_file='dataset/train/train_users_merged_advanced_v8.csv'
    )
    
    print("--- MERGING TRANSACTIONS FOR TEST SET ---")
    merge_transactions(
        v7_file='dataset/test/test_users_merged_advanced_v7.csv',
        transactions_file='dataset/test/test_users_transaction_attempts_v1.csv',
        output_file='dataset/test/test_users_merged_advanced_v8.csv'
    )
