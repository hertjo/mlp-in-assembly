# Unit test for relu, including a call with length 0 that must not touch memory.
#> 0 0 7 0 0 2 5 0
#> -1

.import ../src/sys.s
.import ../src/relu.s

.data
v:     .word -3 0 7 -1 -100 2 5 -8
guard: .word -1

.text
main:
    la a0 v
    li a1 8
    jal relu
    la a0 guard
    li a1 0
    jal relu

    la a0 v
    li a1 1
    li a2 8
    jal print_matrix
    la a0 guard
    li a1 1
    li a2 1
    jal print_matrix
    jal quit
