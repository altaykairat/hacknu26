import csv
import sys
from datetime import datetime

def merge_generations_into_users(users_file, generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Pair ---")
    print(f"Users Data:       {users_file}")
    print(f"Generations Data: {generations_file}")
    
    # 1. Read users file and initialize dictionary mappings
    users_data = {}
    users_headers = []
    
    print(f"[{datetime.now()}] Loading predefined users into memory...")
    with open(users_file, mode='r', encoding='utf-8') as uf:
        reader = csv.DictReader(uf)
        users_headers = list(reader.fieldnames)
        
        for row in reader:
            uid = row.get('user_id')
            if not uid:
                continue
                
            # Store the original row values 
            users_data[uid] = dict(row)
            
            # Initialize our aggregated counters to 0 for EVERY user
            users_data[uid].update({
                'total_generations': 0,
                'completed_generations': 0,
                'failed_generations': 0,
                'nsfw_generations': 0
            })
            
    print(f"[{datetime.now()}] Loaded {len(users_data)} users.")
    
    # 2. Stream through generations file and tally counts
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.DictReader(gf)
        for row in reader:
            uid = row.get('user_id')
            
            # Only update if the user exists in our pre-loaded users database
            if uid in users_data:
                status = row.get('status', '').lower()
                
                users_data[uid]['total_generations'] += 1
                if status == 'completed':
                    users_data[uid]['completed_generations'] += 1
                elif status == 'failed':
                    users_data[uid]['failed_generations'] += 1
                elif status == 'nsfw':
                    users_data[uid]['nsfw_generations'] += 1
                    
            count += 1
            if count % 2000000 == 0:
                print(f"[{datetime.now()}] Processed {count} generation rows...")
                
    print(f"[{datetime.now()}] Finished reading generations. Total generation rows evaluated: {count}.")

    # 3. Write out the enriched dataset
    print(f"[{datetime.now()}] Writing aggregated data to -> {output_file}")
    
    # Define our final columns
    out_headers = users_headers + ['total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        # Write all users (even those with 0 generations will be included neatly)
        for uid, data in users_data.items():
            writer.writerow(data)
            
    print(f"[{datetime.now()}] Successfully saved output!")


if __name__ == '__main__':
    # Execute for Tracking (Train)
    merge_generations_into_users(
        users_file='dataset/train/train_users.csv',
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_users_with_generations.csv'
    )
    
    # Execute for Tracking (Test)
    merge_generations_into_users(
        users_file='dataset/test/test_users.csv',
        generations_file='dataset/test/test_users_generations.csv',
        output_file='dataset/test/test_users_with_generations.csv'
    )
