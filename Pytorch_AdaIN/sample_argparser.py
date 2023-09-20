import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Experiment with DomainNet')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--epoch', '-e', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
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
    parser.add_argument('--resume', type=str)

    return parser.parse_args()
