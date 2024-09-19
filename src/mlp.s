# mlp.s
# The forward pass of the two-layer network.

.globl mlp_forward

.text

# mlp_forward(a0 = W1, a1 = W2, a2 = X, a3 = H, a4 = S)
# All arguments are matrix records (see matio.s). Computes
#     H = relu(W1 * X)
#     S = W2 * H
# allocating H and S and filling in their records. X holds one input per
# column, so a single call handles any number of inputs.
mlp_forward:
    addi sp sp -24
    sw ra 0(sp)
    sw s0 4(sp)
    sw s1 8(sp)
    sw s2 12(sp)
    sw s3 16(sp)
    sw s4 20(sp)

    mv s0 a0
    mv s1 a1
    mv s2 a2
    mv s3 a3
    mv s4 a4

    # H gets rows(W1) x columns(X)
    lw t0 0(s0)
    lw t1 4(s2)
    sw t0 0(s3)
    sw t1 4(s3)
    mul a0 t0 t1
    slli a0 a0 2
    jal alloc
    sw a0 8(s3)

    lw a0 8(s0)             # H = W1 * X
    lw a1 0(s0)
    lw a2 4(s0)
    lw a3 8(s2)
    lw a4 0(s2)
    lw a5 4(s2)
    lw a6 8(s3)
    jal matmul

    lw a0 8(s3)             # H = relu(H)
    lw t0 0(s3)
    lw t1 4(s3)
    mul a1 t0 t1
    jal relu

    # S gets rows(W2) x columns(H)
    lw t0 0(s1)
    lw t1 4(s3)
    sw t0 0(s4)
    sw t1 4(s4)
    mul a0 t0 t1
    slli a0 a0 2
    jal alloc
    sw a0 8(s4)

    lw a0 8(s1)             # S = W2 * H
    lw a1 0(s1)
    lw a2 4(s1)
    lw a3 8(s3)
    lw a4 0(s3)
    lw a5 4(s3)
    lw a6 8(s4)
    jal matmul

    lw ra 0(sp)
    lw s0 4(sp)
    lw s1 8(sp)
    lw s2 12(sp)
    lw s3 16(sp)
    lw s4 20(sp)
    addi sp sp 24
    ret
