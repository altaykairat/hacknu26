import csv
from collections import defaultdict
from datetime import datetime

def aggregate_by_aspect_ratio(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Aspect Ratios ---")
    print(f"Generations Data: {generations_file}")
    
    # Dictionary to keep track of counts per aspect ratio type
    # Structure: aspect_ratio -> {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0}
    ar_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0})
    
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        # Read the header to dynamically find column indices
        headers = next(reader)
        try:
            # NOTE: Checking for 'aspect_ration' since it's misspelled in the actual CSV header
            ar_idx = headers.index('aspect_ration')
            status_idx = headers.index('status')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        # Stream rows
        for row in reader:
            # Skip malformed lines
            if len(row) <= max(ar_idx, status_idx):
                continue
                
            ar = row[ar_idx].strip()
            if not ar:
                ar = "UNKNOWN" # Group empty aspect ratios together
            
            status = row[status_idx].strip().lower()
            
            # Tally metrics
            ar_stats[ar]['total'] += 1
            if status == 'completed':
                ar_stats[ar]['completed'] += 1
            elif status == 'failed':
                ar_stats[ar]['failed'] += 1
            elif status == 'nsfw':
                ar_stats[ar]['nsfw'] += 1
                
            # Log progress
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    # Write out the results
    print(f"[{datetime.now()}] Writing aspect ratio stats to -> {output_file}")
    out_headers = ['aspect_ratio', 'total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for ar, data in ar_stats.items():
            writer.writerow({
                'aspect_ratio': ar,
                'total_generations': data['total'],
                'completed_generations': data['completed'],
                'failed_generations': data['failed'],
                'nsfw_generations': data['nsfw']
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    aggregate_by_aspect_ratio(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_aspect_ratio_stats.csv'
    )
