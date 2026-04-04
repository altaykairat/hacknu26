import csv
from collections import defaultdict
from datetime import datetime

def parse_date(date_str):
    if not date_str:
        return None
    if date_str.startswith('1067-'):
        date_str = '2023-' + date_str[5:]
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def extract_last_generation_date(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Extracting Last Generation Date ---")
    print(f"Generations Data: {generations_file}")
    
    # Structure: user_id -> max datetime object (or None)
    user_max_date = defaultdict(lambda: None)
    
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        headers = next(reader)
        try:
            uid_idx = headers.index('user_id')
            created_idx = headers.index('created_at')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        for row in reader:
            if len(row) <= max(uid_idx, created_idx):
                continue
                
            uid = row[uid_idx].strip()
            created_str = row[created_idx].strip()
            
            if not uid or not created_str:
                continue
                
            created_dt = parse_date(created_str)
            if created_dt:
                current_max = user_max_date[uid]
                if current_max is None or created_dt > current_max:
                    user_max_date[uid] = created_dt
                
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    print(f"[{datetime.now()}] Writing results to -> {output_file}")
    out_headers = ['user_id', 'last_generation_date']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for uid, dt in user_max_date.items():
            dt_str = dt.isoformat() if dt else ""
            writer.writerow({
                'user_id': uid,
                'last_generation_date': dt_str
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    extract_last_generation_date(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_last_generation_date.csv'
    )
    extract_last_generation_date(
        generations_file='dataset/test/test_users_generations.csv',
        output_file='dataset/test/test_last_generation_date.csv'
    )
