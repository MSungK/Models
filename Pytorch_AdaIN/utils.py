import os
import os.path as osp
import torch
import cv2


def save_image(tensor, save_path):
    batch = tensor.shape[0]
    tensor = tensor.to('cpu')
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    for i, image in enumerate(tensor):
        image = image.numpy()
        cv2.imwrite(f'{save_path}_{i}+.png', image)
        if i == 5:
            break
