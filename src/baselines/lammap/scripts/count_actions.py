import os

def count_parentheses_in_file(file_path):
    # Open and read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Count the occurrences of '(' and ')'
    open_parentheses = content.count('(')
    close_parentheses = content.count(')')
    
    action_count = min(open_parentheses, close_parentheses)
    return action_count
    
def count_parentheses_in_logs_folder(logs_folder_path):
    total_pairs = 0

    # Iterate over each folder in the logs directory
    for folder_name in os.listdir(logs_folder_path):
        folder_path = os.path.join(logs_folder_path, folder_name)
        
        # Check if the folder path is a directory
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'combined_plan.py')
            
            # Check if the file exists
            if os.path.isfile(file_path):
                pairs_count = count_parentheses_in_file(file_path)
                print(f"Found {pairs_count} pairs of parentheses in {folder_name}")
                total_pairs += pairs_count

    print(f"\nTotal pairs of parentheses across all log folders: {total_pairs}")

# Example usage:
logs_folder_path = 'logs'
count_parentheses_in_logs_folder(logs_folder_path)
