# argmax.s

.globl argmax, argmax_cols

.text

# a0 = argmax(a0 = array, a1 = length >= 1)
# Position of the largest element. On ties the earliest position wins.
argmax:
    lw t0 0(a0)             # best value so far
    li t1 0                 # its position
    li t2 1                 # position being looked at
am_loop:
    bge t2 a1 am_done
    addi a0 a0 4
    lw t3 0(a0)
    ble t3 t0 am_next       # equal values do not replace the earlier one
    mv t0 t3
    mv t1 t2
am_next:
    addi t2 t2 1
    j am_loop
am_done:
    mv a0 t1
    ret

# argmax_cols(a0 = matrix, a1 = rows >= 1, a2 = columns, a3 = out)
# For every column j of the row-major matrix, writes to out[j] the row of its
# largest element (the topmost one on ties). Used to classify a batch whose
# score vectors are the columns of one matrix.
argmax_cols:
    slli t6 a2 2            # bytes from one row to the next
    li t0 0                 # column j
ac_col:
    beq t0 a2 ac_done
    slli t1 t0 2
    add t1 a0 t1            # walks down column j
    lw t2 0(t1)             # best value so far
    li t3 0                 # its row
    li t4 1                 # row being looked at
ac_row:
    bge t4 a1 ac_store
    add t1 t1 t6
    lw t5 0(t1)
    ble t5 t2 ac_next
    mv t2 t5
    mv t3 t4
ac_next:
    addi t4 t4 1
    j ac_row
ac_store:
    sw t3 0(a3)
    addi a3 a3 4
    addi t0 t0 1
    j ac_col
ac_done:
    ret
