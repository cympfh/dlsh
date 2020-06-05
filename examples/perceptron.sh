#!/bin/bash

PATH=$PATH:../bin/
rm -rf data/
mkdir -p data

# Data
########################

# Input data.
cat <<EOM > data/x
6 2
2 1
2 -1
1 1
1 -1
-1 1
-1 -1
EOM

# Expected output data (regression)
cat <<EOM > data/true_y
6 1
0
0
0.5
0.5
1
1
EOM

# Training
########################
echo "Training"

mkfifo data/grad_mse
mkfifo data/grad_act

for i in `seq 100`; do
    printf "\rEpoch: %04d " $i
    printf "; loss = "
    cat data/x |
        linear --dim 1 -w data/fc --lr 2 -i data/grad_act |
        sigmoid -i data/grad_mse -o data/grad_act |
        mse -t data/true_y -o data/grad_mse --summary average |
        tail -1
    printf "\e[F"
done
echo

echo

# Testing
########################
echo "true_y  pred_y"
paste data/true_y <(cat data/x | linear --dim 1 -w data/fc | sigmoid)
