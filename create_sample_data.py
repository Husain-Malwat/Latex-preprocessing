import csv
import os
import random

# Input and output files
input_csv = "/home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/new_csv_outputs/tokens_7k-15k.csv"
output_csv = "/home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/new_csv_outputs/demo_test_7k-15k.csv"

# Read all rows from the input CSV
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

# Check if we have enough rows
total_rows = len(all_rows)
sample_size = min(48, total_rows)

print(f"Total rows in {input_csv}: {total_rows}")
print(f"Sampling {sample_size} rows...")

# Randomly sample rows
sampled_rows = random.sample(all_rows, sample_size)

# Write sampled rows to output CSV
with open(output_csv, 'w', newline='') as f:
    fieldnames = ['file', 'token_count']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(sampled_rows)

print(f"Created {output_csv} with {len(sampled_rows)} sampled entries")


