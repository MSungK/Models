import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T
import os.path as osp
from PIL import Image
import torch.nn.functional as F


class DomainNet(Dataset):
    def __init__(self, root, num_classes, transform=None):
        assert osp.exists(root) == True
        super().__init__()
        self.transform = transform
        self.root = root
        info = open(root, 'r')
        self.info = info.readlines()
        self.num_classes = num_classes
        info.close()

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_path, label_origin = self.info[idx].split(' ')
        img = Image.open(osp.join(osp.dirname(self.root), img_path))
        if self.transform:
            img = self.transform(img)
        label_origin = torch.tensor(int(label_origin))
        label = F.one_hot(label_origin, num_classes = self.num_classes).type(torch.float64)
        return img, label
    
