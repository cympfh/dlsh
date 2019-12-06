#!/bin/bash

PATH=$PATH:../bin
rm -rf data
mkdir data

cat <<EOM > data/x
6 3
1 -1 -1
-1 1 -1
-1 -1 1
1 0 0
0 1 0
0 0 1
EOM

cat <<EOM > data/y
6 3
1 0 0
0 1 0
0 0 1
0.8 0.1 0.1
0.1 0.8 0.1
0.1 0.1 0.8
EOM

for i in `seq 1`; do
    cat data/x |
        bp linear --dim 3 -w data/fc --lr 2 + activate softmax + kl -t data/y
done

cat data/x |
    linear --dim 3 -w data/fc |
    activate softmax
