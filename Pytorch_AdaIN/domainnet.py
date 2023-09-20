import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T
import os.path as osp
from PIL import Image
import torch.nn.functional as F


source_transforms = T.Compose([
            T.Resize((300, 300)),
            T.RandomCrop((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

target_transforms = T.Compose([
        T.Resize((300, 300)),
        T.CenterCrop((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DomainNet(Dataset):
    def __init__(self, root, train=True):
        assert osp.exists(root) == True
        super().__init__()
        if train == True:
            self.transforms = source_transforms
        else:
            self.transforms = target_transforms
        self.root = root
        info = open(root, 'r')
        self.info = info.readlines()
        info.close()

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_path, label_origin = self.info[idx].split(' ')
        img = Image.open(osp.join(osp.dirname(self.root), img_path))
        if self.transform:
            img = self.transform(img)
        return img
    
