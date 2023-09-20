import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
from torchvision import transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from utils import save_image
from dataset import PreprocessDataset, denorm
from eff_model import Model
from torch.optim.lr_scheduler import LambdaLR
from domainnet import DomainNet


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=5,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--train_content_dir', '-tcd', type=str, default='/workspace/Minsung/dataset/office_31/amazon.txt',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', '-tsd', type=str, default='/workspace/Minsung/dataset/domainnet/clipart.txt',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', '-ttcd', type=str, default='/workspace/Minsung/dataset/domainnet/clipart.txt',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', '-ttsd', type=str, default='/workspace/Minsung/dataset/office_31/dslr.txt',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default='pretrain.pth',
                        help='model state path to load for reuse')

    opt = parser.parse_args()


    print(f'cur content : {opt.train_content_dir}')
    print(f'cur style: {opt.train_style_dir}')
    device = 'cuda:0'

    # create directory to save
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    loss_dir = f'{opt.save_dir}/loss'
    model_state_dir = f'{opt.save_dir}/model_state'
    image_dir = f'{opt.save_dir}/image'

    model = Model().to(device)

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
    
    target_data = DomainNet(opt.train_style_dir, transform=target_transforms)

    source_dataloader = DataLoader(source_data, batch_size=opt.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=opt.workers, drop_last=True)
    target_dataloader = DataLoader(target_data, batch_size=opt.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=opt.workers, drop_last=True)
    
    source_data_test = DomainNet(opt.test_content_dir, transform=source_transforms)
    
    target_data_test = DomainNet(opt.test_style_dir, transform=target_transforms)

    source_dataloader_test = DataLoader(source_data_test, batch_size=opt.batch_size,
                                    shuffle=False, pin_memory=True, num_workers=opt.workers, drop_last=True)
    target_dataloader_test = DataLoader(target_data_test, batch_size=opt.batch_size, 
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
        print(f'epoch:{e:03d}, loss:{loss_list[-1]:03f}')
        
        if e % opt.snapshot_interval == 0:
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
                content = denorm(x_source_test, device)
                style = denorm(x_target_test, device)
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

if __name__ == '__main__':
    main()
