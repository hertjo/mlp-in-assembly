# classify.s
# Classifies a single image.
#
#   classify.s W1 W2 IMAGE SCORES
#
# W1 (128 x 784) and W2 (10 x 128) are the integer weights, IMAGE is a
# 784 x 1 column of pixel values 0..255. The ten output scores are written to
# SCORES and the predicted digit (the position of the largest score) is
# printed on its own line.

.import sys.s
.import matio.s
.import dot.s
.import matmul.s
.import relu.s
.import argmax.s
.import mlp.s

.globl main

.data
w1:     .word 0 0 0
w2:     .word 0 0 0
image:  .word 0 0 0
hidden: .word 0 0 0
scores: .word 0 0 0

.text
main:
    addi sp sp -8
    sw ra 0(sp)
    sw s0 4(sp)

    mv s0 a1                # argv

    lw a0 4(s0)
    la a1 w1
    jal load_matrix
    lw a0 8(s0)
    la a1 w2
    jal load_matrix
    lw a0 12(s0)
    la a1 image
    jal load_matrix

    la a0 w1
    la a1 w2
    la a2 image
    la a3 hidden
    la a4 scores
    jal mlp_forward

    lw a0 16(s0)
    la a1 scores
    jal save_matrix

    la t0 scores
    lw a0 8(t0)
    lw t1 0(t0)
    lw t2 4(t0)
    mul a1 t1 t2
    jal argmax
    jal put_int
    li a0 '\n'
    jal put_char

    lw ra 0(sp)
    lw s0 4(sp)
    addi sp sp 8
    jal quit
