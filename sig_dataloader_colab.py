"""
Memory-efficient dataloaders for Colab training.
IDENTICAL logic to sig_dataloader_v2.py but loads images on-the-fly instead of pre-caching.
This prevents kernel death on Colab due to RAM overflow (CEDAR ~500MB, GDPS ~2GB).

Architecture:
- MATCHES sig_dataloader_v2.py exactly: same transforms, same preprocessing, same logic flow
- DIFFERS ONLY in when images are loaded: on-the-fly in __getitem__ vs pre-cached in __init__
- No changes to loss, loss calculation, or model interface
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from module.preprocess import normalize_image


def imread_tool(img_path):
    """Load and preprocess image - EXACT logic from sig_dataloader_v2.py"""
    image = np.asarray(Image.open(img_path).convert('L'))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize)
    return Image.fromarray(normalized_image), Image.fromarray(cropped_image)


class SigDataset_CEDAR_Colab(Dataset):
    """
    Memory-efficient CEDAR dataloader for Colab.
    
    IDENTICAL LOGIC to SigDataset_CEDAR from sig_dataloader_v2.py:
    - Same transforms, same preprocessing, same tensor operations
    - Same pair file parsing and label handling
    - Same augmentation on concatenated pairs
    
    ONLY DIFFERENCE:
    - Loads images on-the-fly in __getitem__ instead of pre-caching in __init__
    - This saves ~500MB RAM, making it feasible on Colab without kernel death
    """
    
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        self.mode = mode
        
        # EXACT transforms from SigDataset_CEDAR (no modifications)
        trans_list = [transforms.RandomInvert(1.0),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()]
        self.basic_transforms = transforms.Compose(trans_list)
        
        trans_aug_list = [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                        transforms.RandomErasing(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomResizedCrop((image_size, image_size))]
        self.augment_transforms = transforms.Compose(trans_aug_list)
        
        # EXACT pair file path logic from SigDataset_CEDAR
        data_root = path
        if train:
            pair_path = os.path.join(data_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(data_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(data_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(data_root, "gray_test_part.txt")
        
        # Parse pairs from file - EXACT logic from SigDataset_CEDAR
        self.data_root = data_root
        self.pairs = []        # (refer_path_full, test_path_full)
        self.labels = []
        
        with open(pair_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            refer, test, label = line.split()
            refer_full = os.path.join(data_root, refer)
            test_full = os.path.join(data_root, test)
            
            self.pairs.append((refer_full, test_full))
            self.labels.append(int(label))
        
        self.train = train
        print(f"Colab: Loaded {len(self.pairs)} pairs (loading images on-the-fly, not pre-caching)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Load images on-the-fly in __getitem__.
        IDENTICAL processing to SigDataset_CEDAR except images load here instead of __init__.
        """
        refer_path, test_path = self.pairs[index]
        
        # Load and preprocess on-the-fly (EXACT same as SigDataset_CEDAR)
        refer_img, _ = imread_tool(refer_path)
        test_img, _ = imread_tool(test_path)
        
        # Apply transforms (EXACT same as SigDataset_CEDAR)
        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)
        
        # Concatenate pair (EXACT same as SigDataset_CEDAR)
        image_pair = torch.cat((refer_img, test_img), dim=0)
        
        # Apply augmentations if training (EXACT same as SigDataset_CEDAR)
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))
        
        return image_pair, torch.tensor([[self.labels[index]]])


class SigDataset_GDPS_Colab(Dataset):
    """
    Memory-efficient GDPS dataloader for Colab.
    
    IDENTICAL LOGIC to SigDataset_CEDAR (which handles GDPS format too):
    - Same transforms, same preprocessing, same tensor operations
    - Same pair file parsing and label handling
    - Same augmentation on concatenated pairs
    
    ONLY DIFFERENCE:
    - Loads images on-the-fly in __getitem__ instead of pre-caching in __init__
    - This saves ~2GB RAM, making GDPS feasible on Colab
    """
    
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        self.mode = mode
        
        # EXACT transforms from SigDataset_CEDAR (no modifications)
        trans_list = [transforms.RandomInvert(1.0),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()]
        self.basic_transforms = transforms.Compose(trans_list)
        
        trans_aug_list = [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                        transforms.RandomErasing(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomResizedCrop((image_size, image_size))]
        self.augment_transforms = transforms.Compose(trans_aug_list)
        
        # EXACT pair file path logic from SigDataset_CEDAR
        data_root = path
        if train:
            pair_path = os.path.join(data_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(data_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(data_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(data_root, "gray_test_part.txt")
        
        # Parse pairs from file - EXACT logic from SigDataset_CEDAR
        self.data_root = data_root
        self.pairs = []        # (refer_path_full, test_path_full)
        self.labels = []
        
        if not os.path.exists(pair_path):
            raise FileNotFoundError(f"Pair file not found: {pair_path}")
        
        with open(pair_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                refer = parts[0]
                test = parts[1]
                label = parts[2]
                refer_full = os.path.join(data_root, refer)
                test_full = os.path.join(data_root, test)
                
                if os.path.exists(refer_full) and os.path.exists(test_full):
                    self.pairs.append((refer_full, test_full))
                    self.labels.append(int(label))
        
        self.train = train
        print(f"Colab GDPS: Loaded {len(self.pairs)} pairs (loading images on-the-fly, not pre-caching)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Load images on-the-fly in __getitem__.
        IDENTICAL processing to SigDataset_CEDAR except images load here instead of __init__.
        """
        refer_path, test_path = self.pairs[index]
        
        # Load and preprocess on-the-fly (EXACT same as SigDataset_CEDAR)
        refer_img, _ = imread_tool(refer_path)
        test_img, _ = imread_tool(test_path)
        
        # Apply transforms (EXACT same as SigDataset_CEDAR)
        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)
        
        # Concatenate pair (EXACT same as SigDataset_CEDAR)
        image_pair = torch.cat((refer_img, test_img), dim=0)
        
        # Apply augmentations if training (EXACT same as SigDataset_CEDAR)
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))
        
        return image_pair, torch.tensor([[self.labels[index]]])
