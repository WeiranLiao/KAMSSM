#!/bin/sh



python main.py --dataset HW --num_epoch 200 --device 0  --lr 0.01 --weight_decay 5e-4 --delta 0.05 --eta 0.5 --hidden_channels  64

for i in  0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5  ; do
  python main.py --dataset Caltech101-all  --device 2 --ratio $i
done
