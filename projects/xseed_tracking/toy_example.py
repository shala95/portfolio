import json
import os

def split_json_file(file_path, output_directory, num_splits):
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate the size of each split
    total_size = len(data)
    split_size = total_size // num_splits
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Split the data and save each part
    for i in range(num_splits):
        start_index = i * split_size
        # Ensure the last split gets any remaining data
        end_index = (i + 1) * split_size if i != num_splits - 1 else total_size
        split_data = data[start_index:end_index]
        
        # Define the output file path
        output_file_path = os.path.join(output_directory, f'split_{i + 1}.json')
        
        # Save the split data to the output file
        with open(output_file_path, 'w') as outfile:
            json.dump(split_data, outfile, indent=4)
        
        print(f"Saved split {i + 1} to {output_file_path}")

# Example usage
file_path = 'data/player_1/shinguard_left.json'  # Path to your large JSON file
output_directory = 'data/splits'   # Directory where the splits will be saved
num_splits = 1000                # Number of splits

split_json_file(file_path, output_directory, num_splits)