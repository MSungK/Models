import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='ViT with DomainNet')
    parser.add_argument('--train_root', default='/workspace/minsung/dataset/domainnet/quickdraw_train.txt')
    parser.add_argument('--val_root', default='/workspace/minsung/dataset/domainnet/quickdraw_test.txt')
    parser.add_argument('--device', nargs='+', default='cpu')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--save', required=True)
    parser.add_argument('--checkpoint', type=int, default=10)
    return parser.parse_args()