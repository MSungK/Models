import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model
from torch.optim.lr_scheduler import LambdaLR


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=5,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--train_content_dir', '-tcd', type=str, default='../dataset/AdaIN/train_qdr',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', '-tsd', type=str, default='../dataset/AdaIN/train_content',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', '-ttcd', type=str, default='../dataset/AdaIN/train_qdr',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', '-ttsd', type=str, default='../dataset/AdaIN/test_content',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default='pretrain.pth',
                        help='model state path to load for reuse')

    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'
    image_dir = f'{args.save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(args.train_content_dir, args.train_style_dir)
    test_dataset = PreprocessDataset(args.test_content_dir, args.test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_iter = iter(test_loader)
    print(f'Length of test image pairs: {len(test_dataset)}')
    print()

    # set model and optimizer
    model = Model().to(device)
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse), strict=False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    # start training
    loss_list = []
    best_val = 1e100
    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        epoch_loss = list()
        for i, (content, style) in (enumerate(tqdm(train_loader))):
            sleep(1e-2)
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            print(f'[{e}/total {args.epoch} epoch],[{i} /'
                  f'total {round(iters/args.batch_size)} iteration]: {loss.item()}')
        loss_list.append((sum(epoch_loss)/len(epoch_loss)))
        print(f'epoch:{e:03d}, loss:{loss_list[-1]:03f}')

        if e % args.snapshot_interval == 0:
            val_loss = 1e100
            for j, (content, style) in enumerate(test_loader, 1):
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generate(content, style)
                    val_loss = model(content, style).item()
                content = denorm(content, device)
                style = denorm(style, device)
                out = denorm(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch__iteration.png', nrow=args.batch_size)
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
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
