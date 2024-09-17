# Unit test for argmax: a tie (earliest wins), all negative, a single element.
#> 2
#> 1
#> 0

.import ../src/sys.s
.import ../src/argmax.s

.data
tie:    .word 4 -2 9 1 9 0
neg:    .word -5 -3 -7
single: .word 42

.text
main:
    la a0 tie
    li a1 6
    jal argmax
    jal put_int
    li a0 '\n'
    jal put_char

    la a0 neg
    li a1 3
    jal argmax
    jal put_int
    li a0 '\n'
    jal put_char

    la a0 single
    li a1 1
    jal argmax
    jal put_int
    li a0 '\n'
    jal put_char

    jal quit
