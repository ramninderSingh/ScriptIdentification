import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the paths
base_path = '/DATA1/ocrteam/recognition/test'
folders = ['hindi', 'english']
all_subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
other_folders = [folder for folder in all_subfolders if folder not in folders]

def preprocess_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image
            img = img.resize(STANDARD_SIZE, Image.LANCZOS)
            # Convert to numpy array
            return np.array(img)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def get_image_count(folder_path):
    # Count the number of valid image files in a folder
    count = 0
    for root, _, files in os.walk(folder_path):
        count += sum(1 for filename in files if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))
    return count


STANDARD_SIZE = (224, 224)  # This is a common input size for many models

# Get the minimum number of images across all folders
min_image_count = min(get_image_count(os.path.join(base_path, f)) for f in folders)
max_number = 10000  # Set max_number as the minimum image count

# Initialize an empty list to hold dataset information
data = []

# Loop through each folder and its subfolders
for folder in folders:
    folder_path = os.path.join(base_path, folder)

    # Counter to track the number of images processed per category
    count = 0

    # Walk through the directory tree
    for root, _, files in os.walk(folder_path):
        # Use tqdm to show progress
        for filename in tqdm(files, desc=f"Processing {folder}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Check if max_number has been reached for this category
                if count >= max_number:
                    break

                # Full path to the image
                image_path = os.path.join(root, filename)

                # Preprocess the image
                image_array = preprocess_image(image_path)

                if image_array is not None:
                    label = os.path.basename(folder) 
                    data.append({'image': image_array, 'label': label, 'filename': filename})
                    count += 1

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Print some statistics
print(f"Dataset created with {len(df)} entries.")
print(f"Image shape: {df['image'].iloc[0].shape}")
print(f"Labels: {df['label'].value_counts().to_dict()}")

# Create a name for the pickle file based on the first letter of each folder
folder_names = ''.join([f for f in folders])  # Get the first letter of each folder
file_name = f"all_train/Real_test_{folder_names}.pkl"

# Save the DataFrame as a pickle file
df.to_pickle(file_name)
print(f"Dataset saved as '{file_name}'")
