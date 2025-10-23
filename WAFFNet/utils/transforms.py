# utils/transforms.py
import torchvision.transforms as T
from ..config import STANDARD_SIZE

def get_train_transforms():
    return T.Compose([
        T.Resize(STANDARD_SIZE),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.RandomRotation(degrees=5),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        T.RandomPerspective(distortion_scale=0.3, p=0.5),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize(STANDARD_SIZE),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
    ])
