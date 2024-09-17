# argmax.s

.globl argmax

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
