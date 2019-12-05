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

echo "Forwarding"
cat data/x | linear | sigmoid

echo "MSE Loss"
cat data/x | linear | sigmoid | mse -t data/true_y
