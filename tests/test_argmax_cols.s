# Unit test for argmax_cols on a 3x4 matrix:
#   column 0: 1 7 3     -> row 1
#   column 1: -4 -2 -9  -> row 1 (all negative)
#   column 2: 6 6 1     -> row 0 (tie, topmost wins)
#   column 3: 2 0 5     -> row 2
#> 1 1 0 2

.import ../src/sys.s
.import ../src/argmax.s

.data
m:   .word 1 -4 6 2  7 -2 6 0  3 -9 1 5
out: .word 0 0 0 0

.text
main:
    la a0 m
    li a1 3
    li a2 4
    la a3 out
    jal argmax_cols
    la a0 out
    li a1 1
    li a2 4
    jal print_matrix
    jal quit
