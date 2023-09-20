import os
import os.path as osp
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
import os.path as osp
from domainnet import DomainNet
from model import Effi_B7
from argparser import arg_parse
from tqdm import tqdm
from time import sleep
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch.backends.cudnn as cudnn


def main(opt):
    if dist.get_rank() == 0:
        print(f'cur source : {opt.source_root}')
        print(f'cur target : {opt.target_root}')
        print(f'num_classes : {opt.num_classes}')
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.cuda.current_device()

    model = Effi_B7(opt.num_classes).cuda()
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)

    # for name, child in model.named_children():
    #     print(child)
    # exit()

    source_transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    target_transforms = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    source_data = DomainNet(opt.source_root, opt.num_classes, transform=source_transforms)
    source_sampler = DistributedSampler(source_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    target_data = DomainNet(opt.target_root, opt.num_classes, transform=target_transforms)
    target_sampler = DistributedSampler(target_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)


    source_dataloader = DataLoader(source_data, batch_size=opt.batch, sampler=source_sampler,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)
    target_dataloader = DataLoader(target_data, batch_size=opt.batch, sampler=target_sampler,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)

    criterion = nn.CrossEntropyLoss().to(device)
    to_softmax = nn.Softmax(dim=1)
    optimizer = AdamW(params=model.parameters(), lr=opt.lr, weight_decay=1e-6)

    num_steps = max(len(source_dataloader), len(target_dataloader))

    iter_source = iter(source_dataloader)
    iter_target = iter(target_dataloader)

    max_val = 0

    for e in range(opt.epochs):
        source_total = 0
        source_acc = 0
        target_total = 0
        target_acc = 0

        for n_iter in tqdm(range(num_steps)):
            sleep(0.01)
            optimizer.zero_grad()
            model.train()
            try:
                x_source, y_source = next(iter_source)
            except:
                iter_source = iter(source_dataloader)
                x_source, y_source = next(iter_source)
            try:
                x_target, y_target = next(iter_target)
            except:
                iter_target = iter(target_dataloader)
                x_target, y_target = next(iter_target)

            x_source = x_source.to(device)
            y_source = y_source.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            pred_z_s, pred_z_t, pred_z_st, pred_z_ts, projected_z_s, projected_z_t, projected_z_st, projected_z_ts = model(x_source, x_target)
            
            normalized_projected_z_s = projected_z_s / torch.linalg.norm(projected_z_s, dim=1, ord=2, keepdim=True).to(device)
            normalized_projected_z_st = projected_z_st / torch.linalg.norm(projected_z_st, dim=1, ord=2, keepdim=True).to(device)
            similarity_s = torch.matmul(normalized_projected_z_s, normalized_projected_z_st.T)

            normalized_projected_z_t = projected_z_t / torch.linalg.norm(projected_z_t, dim=1, ord=2, keepdim=True).to(device)
            normalized_projected_z_ts = projected_z_ts / torch.linalg.norm(projected_z_ts, dim=1, ord=2, keepdim=True).to(device)
            similarity_t = torch.matmul(normalized_projected_z_t, normalized_projected_z_ts.T)

            similarity_s /= opt.temperature
            similarity_t /= opt.temperature
            similarity_s = to_softmax(similarity_s)
            similarity_t = to_softmax(similarity_t)

            labels_s = torch.mm(y_source, y_source.t()) # positive pairs: same class
            labels_t = torch.eye(opt.batch, dtype=torch.float64).to(device) # positive pairs: identity

            loss_infonce_s = -torch.sum(labels_s * similarity_s, dim=1)
            loss_infonce_t = -torch.sum(labels_t * similarity_t, dim=1)

            normalized_s = torch.sum(labels_s)
            normalized_t = torch.sum(labels_t)

            loss_infonce_s = torch.sum(loss_infonce_s) / normalized_s * opt.lam
            loss_infonce_t = torch.sum(loss_infonce_t) / normalized_t * opt.lam
            
            y_source = (1 - opt.epsilon) * y_source + opt.epsilon / opt.num_classes # label smoothing

            pred_z_s = to_softmax(pred_z_s)
            pred_z_st = to_softmax(pred_z_st)

            pred_z_t = to_softmax(pred_z_t)
            
            loss_cls_s = criterion(pred_z_s, y_source)
            loss_cls_st = criterion(pred_z_st, y_source)

            total_loss = loss_cls_s + loss_cls_st + loss_infonce_s + loss_infonce_t
            
            total_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            
            pred_z_s = torch.argmax(pred_z_s, dim = 1)
            pred_z_st = torch.argmax(pred_z_st, dim = 1)
            pred_z_t = torch.argmax(pred_z_t, dim = 1)
            y_source = torch.argmax(y_source, dim = 1)
            y_target = torch.argmax(y_target, dim = 1)
            
            source_total += y_source.shape[0] * 2
            source_acc += (pred_z_s == y_source).sum().item() + (pred_z_st == y_source).sum().item()
            target_total += y_target.shape[0] * 2
            target_acc += (pred_z_t == y_target).sum().item()
        
        if dist.get_rank() == 0:
            print(f'epoch: {e:03d}, loss: {total_loss.item():.3f}, source_acc: {source_acc/source_total*100:.3f}%, target_acc: {target_acc/target_total*100:.3f}%')

        if dist.get_rank() == 0 and opt.save == True and target_acc/target_total*100 > max_val:
            max_val = target_acc/target_total*100
            os.makedirs(opt.save_path, exist_ok=True)
            torch.save(model.state_dict(), osp.join(opt.save_path, 'max_val.pth'))
            f = open(osp.join(opt.save_path, 'val.txt'), "w")
            f.write(f'max_val : {target_acc/target_total*100} epochs: {e}')
            f.close()

if __name__ == '__main__':
    opt = arg_parse()
    torch.multiprocessing.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl', rank=rank, world_size=num_gpus)
    dist.barrier()
    cudnn.benchmark = True
    main(opt)
        