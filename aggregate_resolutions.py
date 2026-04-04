import csv
from collections import defaultdict
from datetime import datetime

def aggregate_by_resolution(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Resolutions ---")
    print(f"Generations Data: {generations_file}")
    
    # Dictionary to keep track of counts per resolution type
    # Structure: resolution -> {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0}
    res_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0})
    
    print(f"[{datetime.now()}] Streaming generations file (this is fast and memory efficient)...")
    count = 0
    
    # We use csv.reader directly here instead of DictReader because it's approximately 2x faster!
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        # Read the header to dynamically find column indices (safeguard)
        headers = next(reader)
        try:
            res_idx = headers.index('resolution')
            status_idx = headers.index('status')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        # Stream rows
        for row in reader:
            # Skip malformed lines
            if len(row) <= max(res_idx, status_idx):
                continue
                
            res = row[res_idx].strip()
            if not res:
                res = "UNKNOWN" # Group empty resolutions together
            
            status = row[status_idx].strip().lower()
            
            # Tally metrics
            res_stats[res]['total'] += 1
            if status == 'completed':
                res_stats[res]['completed'] += 1
            elif status == 'failed':
                res_stats[res]['failed'] += 1
            elif status == 'nsfw':
                res_stats[res]['nsfw'] += 1
                
            # Log progress
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    # Write out the results
    print(f"[{datetime.now()}] Writing resolution stats to -> {output_file}")
    out_headers = ['resolution', 'total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for res, data in res_stats.items():
            writer.writerow({
                'resolution': res,
                'total_generations': data['total'],
                'completed_generations': data['completed'],
                'failed_generations': data['failed'],
                'nsfw_generations': data['nsfw']
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    aggregate_by_resolution(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_resolutions_stats.csv'
    )
