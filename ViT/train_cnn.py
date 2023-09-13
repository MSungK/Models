import os
import os.path as osp
import torchvision.transforms as T
from domainnet import DomainNet
from torch.utils.data import DataLoader
from vit import ViT
from torch.optim import AdamW
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import tqdm
import time
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from argparser import arg_parse
from utils import save_model
from utils import get_param_size


if __name__ == '__main__':
    opt = arg_parse()

    train_root = opt.train_root
    val_root = opt.val_root
    save_path = opt.save
    checkpoint = opt.checkpoint
    num_workers = 8

    # train_root = '/workspace/minsung/dataset/domainnet/clipart_train.txt'
    # val_root = '/workspace/minsung/dataset/domainnet/clipart_test.txt'
    
    device = opt.device
    multi_gpu = False
    lr = opt.lr
    if device != 'cpu':
        if len(device) == 1:
            device = 'cuda:' + device[0]
        else:
            multi_gpu = True
            device = map(int, device)
    
    print(f'device: {device}')

    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = DomainNet(train_root, 345, transform=train_transforms)
    val_data = DomainNet(val_root, 345, transform=val_transforms)

    # train_sampler = DistributedSampler(train_data)
    # val_sampler = DistributedSampler(val_data)

    batch_size = opt.batch
    train_dataloader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, 
    #                               shuffle=True, sampler=train_sampler)
    # val_dataloader = DataLoader(val_data, batch_size=batch_size, 
    #                               shuffle=False, sampler=val_sampler)

    # from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    max_epochs = opt.epochs
    weight = EfficientNet_B0_Weights
    model = efficientnet_b0(weights = weight)
    #  def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    # net = nn.DataParallel(model, device_ids=device)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 345, bias=True))
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    to_softmax = nn.Softmax(dim=1)
    model = model.to(device)
    best_val_acc = 0

    print(f'param size: {get_param_size(model)}')

    for epoch in range(1, max_epochs+1):
        model.train()
        total = 0
        for n_iter, batch in (enumerate(tqdm.tqdm(train_dataloader))):
            time.sleep(0.01)
            optimizer.zero_grad()
            x_train, y_train = batch
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train)

            pred = to_softmax(pred)
            # pred = torch.argmax(to_softmax(pred), dim=-1)
            loss = criterion(pred, y_train)
            loss.backward()

            optimizer.step()
            scheduler.step()

            pred = torch.argmax(pred, dim=-1)
            y_train = torch.argmax(y_train, dim=-1)
            total += y_train.shape[0]
            acc = (pred == y_train).sum().item()

        print(f'epoch: {epoch:03d}, loss: {loss:.3f}, acc: {acc/total*100:.3f}%')
        
        if epoch % checkpoint == 0:
            total = 0
            for n_iter, batch in (enumerate(tqdm.tqdm(val_dataloader))):
                time.sleep(0.01)
                model.eval()
                x_val, y_val = batch
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                with torch.no_grad():
                    pred = model(x_val)
                    pred = to_softmax(pred)
                    pred = torch.argmax(pred, dim=-1)
                    y_val = torch.argmax(y_val, dim=-1)
                    total += y_val.shape[0]
                    acc = (pred == y_val).sum().item()

            print(f'val acc: {acc/total*100:.3f}') 
            if acc/total*100 >= best_val_acc:
                save_model(model, osp.join(save_path, f'effb0_epoch_{epoch}.pth'))

        
    

    



