# Unit test for dot.
#> -23
#> 26
#> 0

.import ../src/sys.s
.import ../src/dot.s

.data
x: .word 2 -1 4 0 3 5
y: .word 1 3 -2 6 2 -4

.text
main:
    # all six elements, step 1 in both vectors
    la a0 x
    la a1 y
    li a2 6
    li a3 1
    li a4 1
    jal dot
    jal put_int
    li a0 '\n'
    jal put_char

    # x[0], x[2] against y[0], y[3]: 2*1 + 4*6
    la a0 x
    la a1 y
    li a2 2
    li a3 2
    li a4 3
    jal dot
    jal put_int
    li a0 '\n'
    jal put_char

    # empty vectors
    la a0 x
    la a1 y
    li a2 0
    li a3 1
    li a4 1
    jal dot
    jal put_int
    li a0 '\n'
    jal put_char

    jal quit
