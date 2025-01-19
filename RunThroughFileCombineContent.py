import os

def merge_files(input_directory, output_file, file_extension):
    try:
        # Open the output file in write mode
        with open(output_file, 'a', encoding='utf-8') as outfile:  # Use 'a' to append during recursion
            # Iterate over all files and directories in the directory
            for file_name in os.listdir(input_directory):
                file_path = os.path.join(input_directory, file_name)  # Full path
                
                # Check if the item is a directory
                if os.path.isdir(file_path):
                    print(f"Entering directory: {file_path}")
                    merge_files(file_path, output_file, file_extension)  # Recursive call for subdirectory
                
                # Check if the item matches the specified file extension
                elif file_name.endswith(file_extension):
                    print(f"Processing file: {file_path}")
                    
                    # Read the content of the file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # Write the content to the output file
                        outfile.write(f"// --- Start of {file_name} ---\n")
                        outfile.write(content)
                        outfile.write(f"\n// --- End of {file_name} ---\n\n")
        
        print(f"All {file_extension} files in '{input_directory}' have been processed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Get user input for directory, output file, and file extension
input_directory = input("Enter the directory containing the files: ").strip()
output_file = input("Enter the name for the merged output file: ").strip()
file_extension = input("Enter the file extension to search for (e.g., .swift): ").strip()

merge_files(input_directory, output_file, file_extension)
