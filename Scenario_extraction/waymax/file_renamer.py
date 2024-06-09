##############################################################################################
# File that can be used to rename WOMD TFRecord files when only part of the dataset is present
##############################################################################################

import os
import glob

# Define the directory containing the files
directory = # LOCATION OF DIRECTORY CONTAINING WOMD TFRECORD FILES
entries = os.listdir(directory)
files = [file for file in entries if os.path.isfile(os.path.join(directory, file))]
file_count = len(files)
i = 0

# Iterate trough all files in the folder
for filepath in glob.glob(os.path.join(directory, 'training*')):
    
    # Get the current filename from the file path
    filename = os.path.basename(filepath).split('-')

    # Add the new numbers add the end of the file
    file_number = str(i).zfill(5)
    total_number = str(file_count).zfill(5)
    new_filename = filename[0] + '-' + file_number + '-of-' + total_number
    
    # Create the full new file path
    new_file_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(filepath, new_file_path)
    i+=1
print("Done!")