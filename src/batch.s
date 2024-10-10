# batch.s
# Classifies many images in one pass and reports the accuracy.
#
#   batch.s W1 W2 DIGITS LABELS SCORES
#
# DIGITS is a 784 x N matrix with one image per column and LABELS a matrix
# with the N true digits. Since the images are columns, the two matrix
# products of the network process the whole batch at once. The program writes
# the 10 x N scores to SCORES and prints the predictions, the labels and the
# accuracy, truncated to one decimal.

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
digits: .word 0 0 0
labels: .word 0 0 0
hidden: .word 0 0 0
scores: .word 0 0 0
txt_pred:   .asciiz "predictions: "
txt_label:  .asciiz "labels:      "
txt_acc:    .asciiz "accuracy:    "
txt_equals: .asciiz " = "

.text
main:
    addi sp sp -24
    sw ra 0(sp)
    sw s0 4(sp)
    sw s1 8(sp)
    sw s2 12(sp)
    sw s3 16(sp)
    sw s4 20(sp)

    mv s0 a0                # argc
    mv s1 a1                # argv
    li t0 6                 # program name + five paths
    bne s0 t0 bad_args

    lw a0 4(s1)
    la a1 w1
    jal load_matrix
    lw a0 8(s1)
    la a1 w2
    jal load_matrix
    lw a0 12(s1)
    la a1 digits
    jal load_matrix
    lw a0 16(s1)
    la a1 labels
    jal load_matrix

    la t0 digits            # s2 = N, the number of images
    lw s2 4(t0)
    la t0 labels            # one label per image, in any shape
    lw t1 0(t0)
    lw t2 4(t0)
    mul t1 t1 t2
    bne t1 s2 bad_labels

    la a0 w1
    la a1 w2
    la a2 digits
    la a3 hidden
    la a4 scores
    jal mlp_forward

    lw a0 20(s1)
    la a1 scores
    jal save_matrix

    # predictions: argmax of every score column
    slli a0 s2 2
    jal alloc
    mv s3 a0
    la t0 scores
    lw a0 8(t0)
    lw a1 0(t0)
    mv a2 s2
    mv a3 s3
    jal argmax_cols

    la a0 txt_pred
    jal put_str
    mv a0 s3
    li a1 1
    mv a2 s2
    jal print_matrix

    la a0 txt_label
    jal put_str
    la t0 labels
    lw a0 8(t0)
    li a1 1
    mv a2 s2
    jal print_matrix

    # s4 = number of correct predictions
    li s4 0
    la t0 labels
    lw t0 8(t0)             # next label
    mv t1 s3                # next prediction
    mv t2 s2                # images left
count:
    beqz t2 counted
    lw t3 0(t0)
    lw t4 0(t1)
    bne t3 t4 count_next
    addi s4 s4 1
count_next:
    addi t0 t0 4
    addi t1 t1 4
    addi t2 t2 -1
    j count
counted:

    # "accuracy:    C/N = P.D%"
    la a0 txt_acc
    jal put_str
    mv a0 s4
    jal put_int
    li a0 '/'
    jal put_char
    mv a0 s2
    jal put_int
    la a0 txt_equals
    jal put_str
    li t0 1000
    mul t0 s4 t0
    div s3 t0 s2            # accuracy in tenths of a percent
    li t0 10
    div a0 s3 t0
    jal put_int
    li a0 '.'
    jal put_char
    li t0 10
    rem a0 s3 t0
    jal put_int
    li a0 '%'
    jal put_char
    li a0 '\n'
    jal put_char

    lw ra 0(sp)
    lw s0 4(sp)
    lw s1 8(sp)
    lw s2 12(sp)
    lw s3 16(sp)
    lw s4 20(sp)
    addi sp sp 24
    jal quit

bad_args:
    li a0 3
    jal fail

bad_labels:
    li a0 4
    jal fail
