# dot.s

.globl dot

.text

# a0 = dot(a0 = x, a1 = y, a2 = n, a3 = step in x, a4 = step in y)
# Returns the sum of x[i * a3] * y[i * a4] for i = 0 .. n-1.
# Steps count elements, not bytes: step 1 walks along a row of a row-major
# matrix, a step equal to the column count walks down a column.
# Products and the sum wrap around at 32 bits, like int arithmetic in C.
dot:
    slli a3 a3 2            # steps in bytes
    slli a4 a4 2
    li t0 0                 # sum
dot_loop:
    beqz a2 dot_done
    lw t1 0(a0)
    lw t2 0(a1)
    mul t1 t1 t2
    add t0 t0 t1
    add a0 a0 a3
    add a1 a1 a4
    addi a2 a2 -1
    j dot_loop
dot_done:
    mv a0 t0
    ret
