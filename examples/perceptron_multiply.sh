#!/bin/bash

PATH=$PATH:../bin/
rm -rf data/
mkdir -p data

# Data
########################

# Input data.
cat <<EOM > data/x
11 2
0 0
0 1
0 2
1 0
1 1
1 2
1 3
2 0
2 1
2 2
2 3
EOM

# Expected output data (regression)
cat <<EOM > data/true_y
11 1
0
0
0
0
1
2
3
0
2
4
6
EOM

# Training
########################
echo "Training"

mkfifo data/grad_mse
mkfifo data/grad_act

for i in `seq 200`; do
    printf "\rEpoch: %04d" $i
    printf "; lr = %6.4f" $LR
    printf "; loss = "
    LR=$(dc -e "5k 0.7 0.97 $i ^*f")
    cat data/x |
        bp \
            augment noise --var 0.1 +\
            linear --dim 10 -w data/fc --lr $LR +\
            sigmoid +\
            augment noise --var 0.1 +\
            linear --dim 1 -w data/fc2 +\
            mse -t data/true_y -s average |
        tail -1
    printf "\e[F"
done

echo

# Testing
########################
echo "true_y  pred_y"
paste data/true_y \
    <(cat data/x | linear -w data/fc | sigmoid | linear -w data/fc2)
