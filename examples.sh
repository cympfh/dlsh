#!/bin/bash

PATH=$PATH:bin/
mkdir -p data

cat <<EOM > data/x
4 3
1 1 -1
1 -1 1
-1 1 -1
1 -1 -1
EOM

cat <<EOM > data/true_y
4 1
1
1
-1
-1
EOM

rm -f data/fc

echo "Forwarding"
cat data/x | linear --dim 1 -w data/fc | sigmoid

echo "MSE Loss"
cat data/x | linear --dim 1 -w data/fc | sigmoid | mse -t data/true_y
