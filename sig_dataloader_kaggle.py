import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from module.preprocess import normalize_image


def imread_tool(img_path):
    image = np.asarray(Image.open(img_path).convert('L'))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize)
    return Image.fromarray(normalized_image), Image.fromarray(cropped_image)


class SigDataset_CEDAR_Kaggle(Dataset):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized'):
        self.image_root = image_root
        self.pair_root = pair_root
        self.image_size = image_size
        self.mode = mode
        
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
        
        data_root = image_root

        if train:
            pair_path = os.path.join(pair_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(pair_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_test_part.txt")
        
        self.img_dict = {}
        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in tqdm(os.listdir(dir_path)):
                    if img[-4:] == '.png':
                        img_path = os.path.join(dir_path, img)
                        sig_image, _ = imread_tool(img_path)
                        sig_image = self.basic_transforms(sig_image)
                        self.img_dict[img_path] = sig_image
        
        with open(pair_path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            refer, test, label = line.split()
            
            refer_img = self.img_dict[os.path.join(data_root, refer)]
            test_img = self.img_dict[os.path.join(data_root, test)]
            
            refer_test = torch.cat((refer_img, test_img), dim=0)
            self.datas.append(refer_test)
            self.labels.append(int(label))
        
        self.train = train
        print(f"Kaggle: Loaded {len(self.labels)} pairs (pre-cached in RAM)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_pair = self.datas[index]
        
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))

        return image_pair, torch.tensor([[self.labels[index]]])


class SigDataset_GDPS_Kaggle(Dataset):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized'):
        self.image_root = image_root
        self.pair_root = pair_root
        self.image_size = image_size
        self.mode = mode
        
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
        
        # Handle both flat and nested GDPS structures
        data_root = image_root
        pair_path = None
        
        if train:
            if os.path.exists(os.path.join(pair_root, "train", "gray_train.txt")):
                pair_path = os.path.join(pair_root, "train", "gray_train.txt")
                data_root = os.path.join(image_root, "train")
            elif os.path.exists(os.path.join(pair_root, "gray_train.txt")):
                pair_path = os.path.join(pair_root, "gray_train.txt")
                if opt and hasattr(opt, 'part') and opt.part:
                    pair_path = os.path.join(pair_root, "gray_train_part.txt")
        else:
            if os.path.exists(os.path.join(pair_root, "test", "gray_test.txt")):
                pair_path = os.path.join(pair_root, "test", "gray_test.txt")
                data_root = os.path.join(image_root, "test")
            elif os.path.exists(os.path.join(pair_root, "gray_test.txt")):
                pair_path = os.path.join(pair_root, "gray_test.txt")
                if opt and hasattr(opt, 'part') and opt.part:
                    pair_path = os.path.join(pair_root, "gray_test_part.txt")
        
        self.img_dict = {}
        for root, dirs, files in os.walk(data_root):
            for img in tqdm(files, desc=f"Loading {root}"):
                if img[-4:] == '.png':
                    img_path = os.path.join(root, img)
                    sig_image, _ = imread_tool(img_path)
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
        
        if pair_path is None or not os.path.exists(pair_path):
            raise FileNotFoundError(f"Pair file not found")
        
        with open(pair_path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                refer = parts[0]
                test = parts[1]
                label = parts[2]
                refer_img_path = os.path.join(data_root, refer)
                test_img_path = os.path.join(data_root, test)
                
                if refer_img_path in self.img_dict and test_img_path in self.img_dict:
                    refer_img = self.img_dict[refer_img_path]
                    test_img = self.img_dict[test_img_path]
                    refer_test = torch.cat((refer_img, test_img), dim=0)
                    self.datas.append(refer_test)
                    self.labels.append(int(label))
        
        self.train = train
        print(f"Kaggle GDPS: Loaded {len(self.labels)} pairs (pre-cached in RAM)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_pair = self.datas[index]
        
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))

        return image_pair, torch.tensor([[self.labels[index]]])
