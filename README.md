# dlsh

Deep Learning on Shell.


## How?

```console
# A Matrix data as Text

$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 ) > X
$ cat X
3 2
0 0
0 1
1 1

# This is a (3, 2)-matrix.
# The first line contains the size (row, column), and the (real) values are following.
# This matrix also can means 3 instances, each of which has 2 dimentions.

# "Layer"s are commands

# Shell commands can connected via pipe `|`.
# Layers can stacked with `|`.
$ cat X | linear -w data/fc_1
3 5
-5.5869226 2.384049 -3.234392 0.4512289 -5.5657387
-1.0818138 -1.1542585 3.5078242 1.8856971 -1.1181798
1.7649012 3.585418 11.034574 3.6150548 1.7477741

# -w specifies weight matrix file.
# If no file exists, the weights are initialized on forward-propagation,
# and the weight file be created on back-propagation.

# Activations are also layers.
$ cat X | linear -w data/fc_1 | sigmoid
3 5
0.0037325555 0.91560286 0.03789181 0.61093134 0.0038121643
0.25316292 0.2397121 0.97090954 0.86826414 0.24634908
0.8538224 0.9730228 0.9999839 0.97379005 0.8516718

$ cat X | linear -w data/fc_1 | sigmoid | linear -w data/fc_2
3 1
-3.6515105
4.22001
-3.8068254

$ cat X | linear -w data/fc_1 | sigmoid | linear -w data/fc_2 | sigmoid
3 1
0.025295435
0.9855144
0.021735666
```

## Examples

See `examples/` for more details.
