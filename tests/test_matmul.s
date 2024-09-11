# Unit test for matmul: a 2x3 by 3x2 product and a 1x3 by 3x1 product.
#> 19 -8
#> -9 3
#> 32

.import ../src/sys.s
.import ../src/dot.s
.import ../src/matmul.s

.data
a:   .word 1 -2 3 0 4 -1
b:   .word 2 1 -1 0 5 -3
c:   .word 0 0 0 0
row: .word 1 2 3
col: .word 4 5 6
one: .word 0

.text
main:
    la a0 a
    li a1 2
    li a2 3
    la a3 b
    li a4 3
    li a5 2
    la a6 c
    jal matmul
    la a0 c
    li a1 2
    li a2 2
    jal print_matrix

    la a0 row
    li a1 1
    li a2 3
    la a3 col
    li a4 3
    li a5 1
    la a6 one
    jal matmul
    la a0 one
    li a1 1
    li a2 1
    jal print_matrix

    jal quit
