#!/usr/bin/zsh

port=3011
GPUS=4

python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} \
train.py --lr 1e-3 -b 8 -e 100 --save --num_classes 31 --workers 8 --lam 1 