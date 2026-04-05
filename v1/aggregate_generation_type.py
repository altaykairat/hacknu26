import csv
from collections import defaultdict
from datetime import datetime

def aggregate_by_generation_type(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Generation Type ---")
    print(f"Generations Data: {generations_file}")
    
    # Dictionary to keep track of counts per generation_type
    # Structure: generation_type -> {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0}
    type_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0})
    
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        headers = next(reader)
        try:
            type_idx = headers.index('generation_type')
            status_idx = headers.index('status')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        # Stream rows
        for row in reader:
            if len(row) <= max(type_idx, status_idx):
                continue
                
            gen_type = row[type_idx].strip()
            if not gen_type:
                gen_type = "Auto" # Group empty type as "Auto"
            
            status = row[status_idx].strip().lower()
            
            # Tally metrics
            type_stats[gen_type]['total'] += 1
            if status == 'completed':
                type_stats[gen_type]['completed'] += 1
            elif status == 'failed':
                type_stats[gen_type]['failed'] += 1
            elif status == 'nsfw':
                type_stats[gen_type]['nsfw'] += 1
                
            # Log progress
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    # Write out the results
    print(f"[{datetime.now()}] Writing generation type stats to -> {output_file}")
    out_headers = ['generation_type', 'total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for g_type, data in type_stats.items():
            writer.writerow({
                'generation_type': g_type,
                'total_generations': data['total'],
                'completed_generations': data['completed'],
                'failed_generations': data['failed'],
                'nsfw_generations': data['nsfw']
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    aggregate_by_generation_type(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_generation_type_stats.csv'
    )
