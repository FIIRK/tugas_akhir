from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import torch
import os

# transforms
vgg16_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class
class Model1Dataset(Dataset):
    """
    Class for custom datasets.
    Dataset is made using collection of paths from images.
    Image and its corresponding mask are identified trough it's filename.
    Directory would follow this structure:
        image : '...\images\...\imagename.png'
        mask  : '...\masks\...\imagename.png'
    Images and masks are then loaded using PIL.Image.open()
    """
    def __init__(self,
                 image_paths: list[str],
                 classes: list[str],
                 transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        base_image = Image.open(self.image_paths[idx]).convert('RGB')
        base_mask = Image.open(self.image_paths[idx].replace('images', 'masks'))
        
        if self.transform: 
            image = self.transform(base_image)
        else: 
            image = transforms.functional.to_tensor(base_image)

        mask = transforms.functional.to_tensor(base_mask)
        return image, mask

class Model2Dataset(Dataset):
    """
    Class for custom datasets.
    Dataset is made using collection of paths from images.
    Image and its corresponding mask are identified trough it's filename.
    Directory would follow this structure:
        image : '...\images\...\imagename.png'
        mask  : '...\masks\...\imagename.png'
    Images and masks are then loaded using PIL.Image.open()
    """
    def __init__(self,
                 image_paths: list[str],
                 transform=None,
                 augment=None):
        self.image_paths = image_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        base_image  = Image.open(self.image_paths[idx]).convert('RGB')
        base_filter = Image.open(self.image_paths[idx].replace('images', 'masks').replace('post', 'pre'))
        base_mask   = Image.open(self.image_paths[idx].replace('images', 'masks'))

        if self.augment:
            base_image, base_filter, base_mask = self.augment((base_image, base_filter, base_mask))
        if self.transform:
            image = self.transform(base_image)
        else:
            image = transforms.functional.to_tensor(base_image)

        filter = transforms.functional.to_tensor(base_filter) * 255
        image = image * filter

        mask = transforms.functional.to_tensor(base_mask) * 255
        mask = mask.type(torch.LongTensor)
        mask1 = (mask == 1) + (mask == 2) + (mask == 5)
        mask2 = (mask == 3) + (mask == 4)
        mask = torch.cat([mask1, mask2], dim=0).type(torch.FloatTensor)

        return image, mask
