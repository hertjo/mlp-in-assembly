# relu.s

.globl relu

.text

# relu(a0 = array, a1 = length)
# Replaces every negative element with 0, in place.
relu:
    beqz a1 relu_done
relu_loop:
    lw t0 0(a0)
    bgez t0 relu_next
    sw zero 0(a0)
relu_next:
    addi a0 a0 4
    addi a1 a1 -1
    bnez a1 relu_loop
relu_done:
    ret
