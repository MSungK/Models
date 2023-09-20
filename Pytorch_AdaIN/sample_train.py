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
from eff_model import Model
from sample_argparser import arg_parse
from tqdm import tqdm
from time import sleep
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from dataset import denorm
from torchvision.utils import save_image


def main(opt):
    if dist.get_rank() == 0:
        print(f'cur content : {opt.train_content_dir}')
        print(f'cur style: {opt.train_style_dir}')
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.cuda.current_device()

    # create directory to save
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    loss_dir = f'{opt.save_dir}/loss'
    model_state_dir = f'{opt.save_dir}/model_state'
    image_dir = f'{opt.save_dir}/image'

    model = Model().cuda()
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=True)

    model.load_state_dict(torch.load(opt.resume))
    print(f'Model loads {opt.resume}')

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
    
    source_data = DomainNet(opt.train_content_dir, transform=source_transforms)
    source_sampler = DistributedSampler(source_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    target_data = DomainNet(opt.train_style_dir, transform=target_transforms)
    target_sampler = DistributedSampler(target_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)


    source_dataloader = DataLoader(source_data, batch_size=opt.batch_size, sampler=source_sampler,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)
    target_dataloader = DataLoader(target_data, batch_size=opt.batch_size, sampler=target_sampler,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)
    
    source_data_test = DomainNet(opt.test_content_dir, transform=source_transforms)
    source_sampler_test = DistributedSampler(source_data_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    target_data_test = DomainNet(opt.test_style_dir, transform=target_transforms)
    target_sampler_test = DistributedSampler(target_data_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)


    source_dataloader_test = DataLoader(source_data_test, batch_size=opt.batch_size, sampler=source_sampler_test,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)
    target_dataloader_test = DataLoader(target_data_test, batch_size=opt.batch_size, sampler=target_sampler_test,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)

    
    optimizer = AdamW(params=model.parameters(), lr=opt.lr, weight_decay=1e-6)

    num_steps = max(len(source_dataloader), len(target_dataloader))
    num_steps_test = max(len(source_data_test), len(target_dataloader_test))

    iter_source = iter(source_dataloader)
    iter_target = iter(target_dataloader)

    iter_source_test = iter(source_dataloader_test)
    iter_target_test = iter(target_dataloader_test)

    loss_list = []
    best_val = 100

    for e in range(1, opt.epoch + 1):

        if dist.get_rank() == 0:
            print(f'Start {e} epoch')
        epoch_loss = list()

        for n_iter in tqdm(range(num_steps)):
            sleep(1e-2)
            optimizer.zero_grad()
            model.train()
            try:
                x_source = next(iter_source)
            except:
                iter_source = iter(source_dataloader)
                x_source = next(iter_source)
            try:
                x_target = next(iter_target)
            except:
                iter_target = iter(target_dataloader)
                x_target = next(iter_target)

            x_source = x_source.to(device)
            x_target = x_target.to(device)
            loss = model(x_source, x_target)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # if dist.get_rank() == 0:
            #     print(f'[{e}/total {opt.epoch} epoch],[{e} /'
            #             f'n_iter {n_iter}/{num_steps} loss: {loss.item()}')
            
        loss_list.append((sum(epoch_loss)/len(epoch_loss)))
        if dist.get_rank() == 0:
            print(f'epoch:{e:03d}, loss:{loss_list[-1]:03f}')
        
        if dist.get_rank() == 0 and e % opt.snapshot_interval == 0:
            val_loss = 1e100
            for test_iter in tqdm(range(num_steps_test)):
                try: 
                    x_source_test = next(iter_source_test)
                except:
                    iter_source_test = iter(source_dataloader_test)
                    x_source_test = next(iter_source_test)
                try: 
                    x_target_test = next(iter_target_test)
                except:
                    iter_target_test = iter(target_dataloader_test)
                    x_target_test = next(iter_target_test)

                x_source_test = x_source_test.to(device)
                x_target_test = x_target_test.to(device)
                with torch.no_grad():
                    out = model.module.generate(x_source_test, x_target_test)
                    val_loss = model(x_source_test, x_target_test).item()
                x_source_test = denorm(x_source_test, device)
                x_target_test = denorm(x_target_test, device)
                out = denorm(out, device)
                res = torch.cat([x_source_test, x_target_test, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch__iteration.png', nrow=opt.batch_size)
                if val_loss < best_val:
                    torch.save(model.state_dict(), f'{model_state_dir}/best_val.pth')
                break
        
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    if dist.get_rank() == 0:
        print(f'Loss saved in {loss_dir}')
            

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
        