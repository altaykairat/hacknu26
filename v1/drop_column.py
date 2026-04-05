import csv

def drop_column(in_file, out_file, col_to_drop):
    print(f"Processing {in_file} ...")
    with open(in_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = [f for f in reader.fieldnames if f != col_to_drop]
        
        with open(out_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            count = 0
            for row in reader:
                if col_to_drop in row:
                    del row[col_to_drop]
                writer.writerow(row)
                count += 1
    print(f"Saved {count} rows to {out_file} successfully (dropped '{col_to_drop}').\n")

if __name__ == '__main__':
    drop_column(
        in_file='dataset/train/train_users_merged_advanced_v4.csv', 
        out_file='dataset/train/train_users_merged_advanced_v5.csv', 
        col_to_drop='mode_duration'
    )
    
    drop_column(
        in_file='dataset/test/test_users_merged_advanced_v4.csv', 
        out_file='dataset/test/test_users_merged_advanced_v5.csv', 
        col_to_drop='mode_duration'
    )
