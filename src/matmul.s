# matmul.s

.globl matmul

.text

# matmul(a0 = A, a1 = rows of A, a2 = columns of A,
#        a3 = B, a4 = rows of B, a5 = columns of B,
#        a6 = C)
# C = A * B for row-major int32 matrices. C must have room for
# rows(A) * columns(B) words. Ends the program with exit code 2 if
# columns(A) != rows(B).
#
# C[i][j] is the dot product of row i of A (step 1) and column j of B
# (step columns(B)); C is filled in memory order.
matmul:
    bne a2 a4 mm_shape

    addi sp sp -36
    sw ra 0(sp)
    sw s0 4(sp)
    sw s1 8(sp)
    sw s2 12(sp)
    sw s3 16(sp)
    sw s4 20(sp)
    sw s5 24(sp)
    sw s6 28(sp)
    sw s7 32(sp)

    mv s0 a0                # start of the current row of A
    mv s1 a1                # rows of A still to do
    mv s2 a2                # length of every dot product
    mv s3 a3                # B
    mv s4 a5                # columns of B
    mv s5 a6                # next element of C
    slli s7 a2 2            # bytes per row of A

mm_row:
    beqz s1 mm_done
    li s6 0                 # column j of B
mm_col:
    beq s6 s4 mm_row_done
    mv a0 s0
    slli t0 s6 2
    add a1 s3 t0            # top of column j
    mv a2 s2
    li a3 1
    mv a4 s4
    jal dot
    sw a0 0(s5)
    addi s5 s5 4
    addi s6 s6 1
    j mm_col
mm_row_done:
    add s0 s0 s7
    addi s1 s1 -1
    j mm_row

mm_done:
    lw ra 0(sp)
    lw s0 4(sp)
    lw s1 8(sp)
    lw s2 12(sp)
    lw s3 16(sp)
    lw s4 20(sp)
    lw s5 24(sp)
    lw s6 28(sp)
    lw s7 32(sp)
    addi sp sp 36
    ret

mm_shape:
    li a0 2
    jal fail
