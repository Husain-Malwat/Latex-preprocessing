#!/bin/bash

# Base directories
INPUT_BASE="/home/husainmalwat/workspace/OCR_Latex/data/final_merged"
OUTPUT_BASE="/home/husainmalwat/workspace/OCR_Latex/data/preprocess_1"
BIB_OUTPUT="/home/husainmalwat/workspace/OCR_Latex/data/bibliographies"

# Create bibliography output directory
mkdir -p "$BIB_OUTPUT"

# Loop through years 2000-2023
for year in {2000..2023}; do
    # Determine month range
    if [ $year -eq 2023 ]; then
        months=(01 02 03 04 05 06 07 08)
    else
        months=(01 02 03 04 05 06 07 08 09 10 11 12)
    fi
    
    # Get last two digits of year
    year_short=$(printf "%02d" $((year % 100)))
    
    # Loop through months
    for month in "${months[@]}"; do
        month_str="${year_short}${month}"
        
        # Define paths
        input_dir="${INPUT_BASE}/${year}/final_merged/${month_str}"
        output_dir="${OUTPUT_BASE}/${year}/${month_str}"
        
        # Check if input directory exists
        if [ -d "$input_dir" ]; then
            echo "Processing: $input_dir -> $output_dir"
            
            # Create output directory
            mkdir -p "$output_dir"
            
            # Run preprocessor
            python run_preprocessor.py "$input_dir" -o "$output_dir" --bib-output-dir "$BIB_OUTPUT"
            
            echo "Completed: ${year}/${month_str}"
            echo "---"
        else
            echo "Skipping (not found): $input_dir"
        fi
    done
done

echo "All preprocessing completed!"