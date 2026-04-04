import csv
from collections import defaultdict
from datetime import datetime

def aggregate_by_credit_cost(generations_file, output_file):
    print(f"\n[{datetime.now()}] --- Processing Credit Cost ---")
    print(f"Generations Data: {generations_file}")
    
    # Dictionary to keep track of counts per credit cost
    # Structure: credit_cost -> {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0}
    cost_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'failed': 0, 'nsfw': 0})
    
    print(f"[{datetime.now()}] Streaming generations file...")
    count = 0
    
    with open(generations_file, mode='r', encoding='utf-8') as gf:
        reader = csv.reader(gf)
        
        headers = next(reader)
        try:
            cost_idx = headers.index('credit_cost')
            status_idx = headers.index('status')
        except ValueError as e:
            print(f"Error: Could not find required columns in header: {headers}")
            return
            
        # Stream rows
        for row in reader:
            if len(row) <= max(cost_idx, status_idx):
                continue
                
            cost = row[cost_idx].strip()
            if not cost:
                cost = "Auto" # Group empty cost as "Auto"
            
            status = row[status_idx].strip().lower()
            
            # Tally metrics
            cost_stats[cost]['total'] += 1
            if status == 'completed':
                cost_stats[cost]['completed'] += 1
            elif status == 'failed':
                cost_stats[cost]['failed'] += 1
            elif status == 'nsfw':
                cost_stats[cost]['nsfw'] += 1
                
            # Log progress
            count += 1
            if count % 5000000 == 0:
                print(f"[{datetime.now()}] Processed {count} rows...")
                
    print(f"[{datetime.now()}] Finished extracting data. Evaluated {count} total generation rows.")

    # Write out the results
    print(f"[{datetime.now()}] Writing credit cost stats to -> {output_file}")
    out_headers = ['credit_cost', 'total_generations', 'completed_generations', 'failed_generations', 'nsfw_generations']
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_headers)
        writer.writeheader()
        
        for cost, data in cost_stats.items():
            writer.writerow({
                'credit_cost': cost,
                'total_generations': data['total'],
                'completed_generations': data['completed'],
                'failed_generations': data['failed'],
                'nsfw_generations': data['nsfw']
            })
            
    print(f"[{datetime.now()}] Successfully saved output!")

if __name__ == '__main__':
    aggregate_by_credit_cost(
        generations_file='dataset/train/train_users_generations.csv',
        output_file='dataset/train/train_credit_cost_stats.csv'
    )
