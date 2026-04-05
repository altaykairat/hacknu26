import csv
from collections import defaultdict
from datetime import datetime

def aggregate_by_duration(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Duration ---")
    print(f"Generations Data: {generations_file}")
    
    # Dictionary to keep track of counts per duration
    # Structure: duration -> {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0}
    dur_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0})
    
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        # Read the header to dynamically find column indices
        headers = next(reader)
        try:
            dur_idx = headers.index('duration')
            status_idx = headers.index('status')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        # Stream rows
        for row in reader:
            # Skip malformed lines
            if len(row) <= max(dur_idx, status_idx):
                continue
                
            dur = row[dur_idx].strip()
            if not dur:
                dur = "Auto" # Group empty durations as "Auto"
            
            status = row[status_idx].strip().lower()
            
            # Tally metrics
            dur_stats[dur]['total'] += 1
            if status == 'completed':
                dur_stats[dur]['completed'] += 1
            elif status == 'failed':
                dur_stats[dur]['failed'] += 1
            elif status == 'nsfw':
                dur_stats[dur]['nsfw'] += 1
                
            # Log progress
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    # Write out the results
    print(f"[{datetime.now()}] Writing duration stats to -> {output_file}")
    out_headers = ['duration', 'total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for dur, data in dur_stats.items():
            writer.writerow({
                'duration': dur,
                'total_generations': data['total'],
                'completed_generations': data['completed'],
                'failed_generations': data['failed'],
                'nsfw_generations': data['nsfw']
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    aggregate_by_duration(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_duration_stats.csv'
    )
