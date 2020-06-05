#!/bin/bash

#
# Soliving XOR with 2-layer Perceptoron
#
# XOR: 1 + 1 = 0 + 0 = 0, 1 + 0 = 0 + 1 = 1
#

PATH=$PATH:../bin/
rm -rf data/
mkdir -p data

# Data
########################

cat <<EOM > data/x
8 3
1 1
-1 -1
1 -1
-1 1
2 2
-2 -2
2 -2
-2 2
EOM

cat <<EOM > data/true_y
8 1
0
0
1
1
0
0
1
1
EOM

# Training
########################
echo "Training"

mkfifo data/grad_mse
mkfifo data/grad_act_1
mkfifo data/grad_act_2
mkfifo data/grad_fc_2

average() {
    awk c++ | jq -s '. | add / length' | tr -d '\n'
}

for i in `seq 100`; do
    printf "\rEpoch %04d: " $i
    LR=$( echo 5 0.99 $i | awk '{print $1 * $2 ^ $3}' )
    printf "; lr = %.4f" $LR
    printf "; loss = "
    cat data/x |
        bp \
            augment gaussnoise --var 0.3 +\
            linear --dim 5 -w data/fc_1 --lr $LR +\
            augment gaussnoise --var 0.1 +\
            sigmoid +\
            linear --dim 1 -w data/fc_2 --lr $LR +\
            augment gaussnoise --var 0.01 +\
            sigmoid +\
            mse -t data/true_y |
        average
    printf "\e[K"
done
echo

# Testing
########################

# result on train data
cat data/x | linear -w data/fc_1 | sigmoid | linear -w data/fc_2 | sigmoid

# test on unknown data
(
echo 121 2
for i in $(seq -5 5); do
    for j in $(seq -5 5); do
        echo $i $j
    done
done
) > data/x_test
cat data/x_test |
    linear -w data/fc_1 |
    sigmoid |
    linear -w data/fc_2 | sigmoid > data/y_pred
paste -d ' ' data/x_test data/y_pred | awk c++ > data/test_result

gnuplot -persist <<EOM
set terminal qt
set title 'predicted XOR'
splot 'data/test_result'
pause -1
EOM
