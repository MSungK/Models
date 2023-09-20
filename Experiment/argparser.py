import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Experiment with DomainNet')
    parser.add_argument('--source_root', default='/workspace/Minsung/dataset/office_31/amazon.txt')
    parser.add_argument('--target_root', default='/workspace/Minsung/dataset/office_31/dslr.txt')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch', '-b', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, required=True)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--num_classes', type=int, default=31)
    parser.add_argument('--epsilon', type=float, default=0.1, help='for label smoothing')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--save_path', default='runs/')

    return parser.parse_args()