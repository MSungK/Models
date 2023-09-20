#!/usr/bin/zsh

port=3011
GPUS=4

python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} \
sample_train.py --lr 1e-3 -b 10 -e 100 --workers 8 --resume result/model_state/best_val.pth