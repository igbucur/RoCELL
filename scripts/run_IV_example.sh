# Example of instrumental variable model with the following parameters:
# - number of live points == 500
# - spike-and-slab mixture weight for causal effect (sb32) == 0.5
# - slab precision == 1
# - spike precision == 100
# - covariance matrix == [[1, 1, 1], [1, 2, 2], [1, 2, 3]]
./RoCELL -o example_RoCELL -n 500 -w 0.5 -slab 1 -spike 100 -s11 1 -s12 1 -s13 1 -s22 2 -s23 2 -s33 3
