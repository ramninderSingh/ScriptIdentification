import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the paths
base_path = 'ScriptDataset/TrainDataset/SynData'
folders = ['hindi', 'english']

# Define standard size for all images
STANDARD_SIZE = (224, 224)  # This is a common input size for many models

# Initialize an empty list to hold dataset information
data = []

# Function to preprocess image
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

# Parameter to limit the number of images per category
max_number = 100000  # You can set this to any desired value

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
                    # Append the image data and its corresponding label
                    label = os.path.basename(folder)  # Use the subfolder name as the label
                    data.append({'image': image_array, 'label': label})
                    count += 1

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Print some statistics
print(f"Dataset created with {len(df)} entries.")
print(f"Image shape: {df['image'].iloc[0].shape}")
print(f"Labels: {df['label'].value_counts().to_dict()}")

# Save the DataFrame as a pickle file
df.to_pickle('Syn_train_HE.pkl')
print("Dataset saved as 'Syn_train_HE.pkl'")