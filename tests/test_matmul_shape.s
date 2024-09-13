# matmul must refuse a 2x3 by 2x3 product and exit with code 2.
#! exit 2

.import ../src/sys.s
.import ../src/dot.s
.import ../src/matmul.s

.data
a: .word 1 2 3 4 5 6
c: .word 0 0 0 0 0 0 0 0 0

.text
main:
    la a0 a
    li a1 2
    li a2 3
    la a3 a
    li a4 2
    li a5 3
    la a6 c
    jal matmul
    jal quit
