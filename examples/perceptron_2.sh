#!/bin/bash

PATH=$PATH:../bin/
rm -rf data/
mkdir -p data

# Data:
#
# XOR: 1 + 1 = 0 + 0 = 0, 1 + 0 = 0 + 1 = 1
#
########################

# The last 1 is for bias
cat <<EOM > data/x
4 3
1 1
0 0
1 0
0 1
EOM

cat <<EOM > data/true_y
4 1
0
0
1
1
EOM

echo "Training"
mkfifo data/grad_mse
mkfifo data/grad_act_1
mkfifo data/grad_act_2
mkfifo data/grad_fc_2

for i in `seq 1000`; do
    echo -n "Epoch ${i}: "
    cat data/x |
        linear --dim 5 -w data/fc_1 --lr 5 -i data/grad_act_1 |
        sigmoid -i data/grad_fc_2 -o data/grad_act_1 |
        linear --dim 1 -w data/fc_2 --lr 2 -i data/grad_act_2 -o data/grad_fc_2 |
        sigmoid -i data/grad_mse -o data/grad_act_2 |
        mse -t data/true_y -o data/grad_mse |
        tail -1
done

echo

echo "Forwarding"
cat data/x | linear --dim 2 -w data/fc_1 | sigmoid | linear --dim 1 -w data/fc_2 | sigmoid
