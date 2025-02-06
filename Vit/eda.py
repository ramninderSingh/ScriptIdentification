from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
from torchvision import transforms

def TrainEDA(folder, label, max_images=None):
    images = []
    count = 0  # Counter for loaded images
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
            img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
            images.append({'image': img, 'label': label})
            count += 1
            if max_images is not None and count >= max_images:
                break 
    return images


class TestEDA(Dataset):
    def __init__(self, data_folders,max_images=None):
        self.image_paths = []
        self.labels = []
        self.label_mapping = {label: idx for idx, label in enumerate(data_folders.keys())}
       

        image_count = 0  
        
        for label, folder in data_folders.items():
            folder_image_count = 0 
            
            for image_file in os.listdir(folder):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder, image_file))
                    self.labels.append(self.label_mapping[label])
                    folder_image_count += 1
                    
                    if max_images is not None and folder_image_count >= max_images:
                        break

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to fit the model input size
            transforms.ToTensor(),           # Convert PIL images to tensors
        ])
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label = self.labels[idx]
        image = self.transform(image)
        return image, label