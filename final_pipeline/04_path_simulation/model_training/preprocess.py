import os
import json
import glob
from pathlib import Path

def filter_data_files(input_folder, output_folder):
    """
    Filter out cases where box_collision=true, ground_collision=false, pedestal_collision=false
    from all JSON files in the input folder and save filtered data to output folder.
    
    Args:
        input_folder (str): Path to folder containing original data files
        output_folder (str): Path to folder where filtered data will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "data_*.json"))
    
    print(f"Found {len(json_files)} files to process")
    
    total_cases_before = 0
    total_cases_after = 0
    removed_cases = 0
    
    for file_path in json_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Count cases before filtering
        cases_in_file = len(data)
        total_cases_before += cases_in_file
        
        # Create filtered data dictionary
        filtered_data = {}
        
        # Process each case in the file
        for case_id, case_data in data.items():
            # Check if this case should be filtered out
            should_remove = (
                case_data.get('box_collision', False) == True and
                case_data.get('ground_collision', False) == False and
                case_data.get('pedestal_collision', False) == False
            )
            
            if should_remove:
                removed_cases += 1
                print(f"  Removed case {case_id}: box_collision=True, others=False")
            else:
                filtered_data[case_id] = case_data
        
        # Count cases after filtering
        cases_after_filtering = len(filtered_data)
        total_cases_after += cases_after_filtering
        
        # Save filtered data to output folder
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f)
        
        print(f"  Before: {cases_in_file} cases, After: {cases_after_filtering} cases, Removed: {cases_in_file - cases_after_filtering} cases")
    
    print(f"\n" + "="*60)
    print(f"FILTERING SUMMARY:")
    print(f"="*60)
    print(f"Total cases BEFORE filtering: {total_cases_before:,}")
    print(f"Total cases AFTER filtering:  {total_cases_after:,}")
    print(f"Cases removed:               {removed_cases:,}")
    print(f"Removal percentage:          {(removed_cases/total_cases_before)*100:.2f}%")
    print(f"Retention percentage:        {(total_cases_after/total_cases_before)*100:.2f}%")
    print(f"Filtered files saved to:     {output_folder}")
    print(f"="*60)

def analyze_filtering_criteria(input_folder):
    """
    Analyze the data to understand the distribution of collision patterns.
    
    Args:
        input_folder (str): Path to folder containing data files
    """
    
    json_files = glob.glob(os.path.join(input_folder, "data_*.json"))
    
    # Count different collision patterns
    collision_patterns = {
        'box_only': 0,      # box=True, ground=False, pedestal=False
        'ground_only': 0,   # box=False, ground=True, pedestal=False
        'pedestal_only': 0, # box=False, ground=False, pedestal=True
        'box_ground': 0,    # box=True, ground=True, pedestal=False
        'box_pedestal': 0,  # box=True, ground=False, pedestal=True
        'ground_pedestal': 0, # box=False, ground=True, pedestal=True
        'all_collisions': 0, # box=True, ground=True, pedestal=True
        'no_collisions': 0,  # box=False, ground=False, pedestal=False
        'other': 0          # other combinations
    }
    
    total_cases = 0
    
    for file_path in json_files[:5]:  # Analyze first 5 files for statistics
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for case_id, case_data in data.items():
            total_cases += 1
            
            box = case_data.get('box_collision', False)
            ground = case_data.get('ground_collision', False)
            pedestal = case_data.get('pedestal_collision', False)
            
            if box and not ground and not pedestal:
                collision_patterns['box_only'] += 1
            elif not box and ground and not pedestal:
                collision_patterns['ground_only'] += 1
            elif not box and not ground and pedestal:
                collision_patterns['pedestal_only'] += 1
            elif box and ground and not pedestal:
                collision_patterns['box_ground'] += 1
            elif box and not ground and pedestal:
                collision_patterns['box_pedestal'] += 1
            elif not box and ground and pedestal:
                collision_patterns['ground_pedestal'] += 1
            elif box and ground and pedestal:
                collision_patterns['all_collisions'] += 1
            elif not box and not ground and not pedestal:
                collision_patterns['no_collisions'] += 1
            else:
                collision_patterns['other'] += 1
    
    print(f"\nCollision Pattern Analysis (from {total_cases} cases):")
    for pattern, count in collision_patterns.items():
        percentage = (count / total_cases) * 100
        print(f"  {pattern}: {count} cases ({percentage:.2f}%)")

def count_total_cases(folder_path):
    """
    Count total number of cases in all JSON files in a folder.
    
    Args:
        folder_path (str): Path to folder containing data files
    
    Returns:
        int: Total number of cases
    """
    json_files = glob.glob(os.path.join(folder_path, "data_*.json"))
    total_cases = 0
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        total_cases += len(data)
    
    return total_cases

if __name__ == "__main__":
    # Define paths
    input_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/raw_data"
    output_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/filtered_data"
    
    # Count total cases before any processing
    print("Counting total cases in original data...")
    total_original_cases = count_total_cases(input_folder)
    print(f"Total cases in original data: {total_original_cases:,}")
    
    # First, analyze the data to understand the distribution
    print("\nAnalyzing collision patterns...")
    analyze_filtering_criteria(input_folder)
    
    # Then perform the filtering
    print("\n" + "="*50)
    print("Starting data filtering...")
    filter_data_files(input_folder, output_folder)
    
    # Count total cases after filtering
    print("\nCounting total cases in filtered data...")
    total_filtered_cases = count_total_cases(output_folder)
    print(f"Total cases in filtered data: {total_filtered_cases:,}")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"FINAL SUMMARY:")
    print(f"="*60)
    print(f"Original data cases:  {total_original_cases:,}")
    print(f"Filtered data cases:  {total_filtered_cases:,}")
    print(f"Cases removed:        {total_original_cases - total_filtered_cases:,}")
    print(f"Removal percentage:   {((total_original_cases - total_filtered_cases)/total_original_cases)*100:.2f}%")
    print(f"Retention percentage: {(total_filtered_cases/total_original_cases)*100:.2f}%")
    print(f"="*60)
    
    print("\nFiltering complete!")