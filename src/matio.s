# matio.s
# Matrix files. A file holds two little-endian int32 values, rows and columns,
# followed by rows * columns int32 values, one row after the other.
#
# In memory a matrix is described by a 12-byte record:
#   0(rec)  rows
#   4(rec)  columns
#   8(rec)  address of the first element
# The first eight bytes match the file header, so the header is read straight
# into the record.

.globl load_matrix

.text

# load_matrix(a0 = path, a1 = record)
# Reads the header into the record, allocates room for the elements, reads
# them and stores their address at 8(record). Any failed or short file
# operation ends the program with exit code 1.
load_matrix:
    addi sp sp -20
    sw ra 0(sp)
    sw s0 4(sp)
    sw s1 8(sp)
    sw s2 12(sp)
    sw s3 16(sp)

    mv s1 a1                # record
    li a1 0                 # open for reading
    jal file_open
    li t0 -1
    beq a0 t0 lm_error
    mv s0 a0                # descriptor

    mv a0 s0                # header -> rows and columns of the record
    mv a1 s1
    li a2 8
    jal file_read
    li t0 8
    bne a0 t0 lm_error

    lw t0 0(s1)
    lw t1 4(s1)
    mul s2 t0 t1
    slli s2 s2 2            # payload size in bytes
    mv a0 s2
    jal alloc
    mv s3 a0
    sw s3 8(s1)

    mv a0 s0
    mv a1 s3
    mv a2 s2
    jal file_read
    bne a0 s2 lm_error

    mv a0 s0
    jal file_close
    bnez a0 lm_error

    lw ra 0(sp)
    lw s0 4(sp)
    lw s1 8(sp)
    lw s2 12(sp)
    lw s3 16(sp)
    addi sp sp 20
    ret

lm_error:
    li a0 1
    jal fail
