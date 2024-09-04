# load_matrix and save_matrix. Run from the repository root. Loads
# tests/data/small.bin, prints its shape and elements, writes it to
# tests/data/roundtrip.bin, loads that copy and prints it again.
#> 2 3
#> 5 -7 11
#> 0 2 -3
#> 2 3
#> 5 -7 11
#> 0 2 -3

.import ../src/sys.s
.import ../src/matio.s

.data
src_path:  .asciiz "tests/data/small.bin"
copy_path: .asciiz "tests/data/roundtrip.bin"
first:  .word 0 0 0
second: .word 0 0 0

.text
main:
    la a0 src_path
    la a1 first
    jal load_matrix
    la a0 first
    jal show

    la a0 copy_path
    la a1 first
    jal save_matrix

    la a0 copy_path
    la a1 second
    jal load_matrix
    la a0 second
    jal show
    jal quit

# show(a0 = record): prints "rows cols", then the elements.
show:
    addi sp sp -8
    sw ra 0(sp)
    sw s0 4(sp)
    mv s0 a0
    mv a0 s0                # rows and cols are the first two words
    li a1 1
    li a2 2
    jal print_matrix
    lw a0 8(s0)
    lw a1 0(s0)
    lw a2 4(s0)
    jal print_matrix
    lw ra 0(sp)
    lw s0 4(sp)
    addi sp sp 8
    ret
