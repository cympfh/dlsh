# dlsh

Deep Learning on Shell.

See examples/

## Example

```console
$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 )
3 2
0 0
0 1
1 1

# This data means a matrix with shape (3, 2)
# or 3 instances with 2 dimentions

$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 ) | linear -w data/fc_1
3 5
-5.5869226 2.384049 -3.234392 0.4512289 -5.5657387
-1.0818138 -1.1542585 3.5078242 1.8856971 -1.1181798
1.7649012 3.585418 11.034574 3.6150548 1.7477741

$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 ) | linear -w data/fc_1 | sigmoid
3 5
0.0037325555 0.91560286 0.03789181 0.61093134 0.0038121643
0.25316292 0.2397121 0.97090954 0.86826414 0.24634908
0.8538224 0.9730228 0.9999839 0.97379005 0.8516718

$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 ) | linear -w data/fc_1 | sigmoid | linear -w data/fc_2
3 1
-3.6515105
4.22001
-3.8068254

$ ( echo 3 2; echo 0 0; echo 0 1; echo 1 1 ) | linear -w data/fc_1 | sigmoid | linear -w data/fc_2 | sigmoid
3 1
0.025295435
0.9855144
0.021735666
```
