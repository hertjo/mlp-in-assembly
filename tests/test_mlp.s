# mlp_forward on a tiny network: 2 inputs, 3 hidden units, 2 outputs, and a
# batch of two input columns. Prints H = relu(W1 X), then S = W2 H.
#> 2 0
#> 6 2
#> 0 3
#> 2 -6
#> 18 9

.import ../src/sys.s
.import ../src/dot.s
.import ../src/matmul.s
.import ../src/relu.s
.import ../src/mlp.s

.data
w1_data: .word 1 -1  2 0  -1 1
w2_data: .word 1 0 -2  0 3 1
x_data:  .word 3 1  1 4
w1: .word 3 2 0
w2: .word 2 3 0
x:  .word 2 2 0
h:  .word 0 0 0
s:  .word 0 0 0

.text
main:
    la t0 w1                # point the records at their elements
    la t1 w1_data
    sw t1 8(t0)
    la t0 w2
    la t1 w2_data
    sw t1 8(t0)
    la t0 x
    la t1 x_data
    sw t1 8(t0)

    la a0 w1
    la a1 w2
    la a2 x
    la a3 h
    la a4 s
    jal mlp_forward

    la t0 h
    lw a0 8(t0)
    lw a1 0(t0)
    lw a2 4(t0)
    jal print_matrix
    la t0 s
    lw a0 8(t0)
    lw a1 0(t0)
    lw a2 4(t0)
    jal print_matrix
    jal quit
