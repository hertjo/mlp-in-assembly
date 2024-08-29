# sys.s
# Environment calls of the Venus RISC-V simulator, wrapped so that they follow
# the normal calling convention (arguments from a0 on, result in a0). The
# simulator wants the call number in a0 and the arguments from a1 on, so each
# wrapper moves its arguments up by one register before the ecall.

.globl put_int, put_char, put_str, alloc, quit, fail
.globl file_open, file_read, file_write, file_close, print_matrix

.text

# put_int(a0 = value): prints a signed decimal number.
put_int:
    mv a1 a0
    li a0 1
    ecall
    ret

# put_char(a0 = character code)
put_char:
    mv a1 a0
    li a0 11
    ecall
    ret

# put_str(a0 = address of a zero-terminated string)
put_str:
    mv a1 a0
    li a0 4
    ecall
    ret

# a0 = alloc(a0 = bytes)
# Moves the heap break up and returns the old break. Nothing is ever freed;
# every program here runs once and exits.
alloc:
    mv a1 a0
    li a0 9
    ecall
    ret

# quit(): ends the program with exit code 0. Does not return.
quit:
    li a0 10
    ecall

# fail(a0 = exit code): ends the program with that code. Does not return.
fail:
    mv a1 a0
    li a0 17
    ecall

# a0 = file_open(a0 = path, a1 = mode)
# Returns a descriptor or -1. Mode 0 opens the file for reading, mode 1
# creates or truncates it for writing.
file_open:
    mv a2 a1
    mv a1 a0
    li a0 13
    ecall
    ret

# a0 = file_read(a0 = descriptor, a1 = buffer, a2 = byte count)
# Returns the number of bytes actually read.
file_read:
    mv a3 a2
    mv a2 a1
    mv a1 a0
    li a0 14
    ecall
    ret

# a0 = file_write(a0 = descriptor, a1 = buffer, a2 = word count)
# Writes a2 32-bit words and returns how many were written.
file_write:
    mv a3 a2
    mv a2 a1
    mv a1 a0
    li a4 4
    li a0 15
    ecall
    ret

# a0 = file_close(a0 = descriptor): 0 on success, -1 on failure.
file_close:
    mv a1 a0
    li a0 16
    ecall
    ret

# print_matrix(a0 = first element, a1 = rows, a2 = columns)
# One line per row, values separated by single spaces.
print_matrix:
    addi sp sp -20
    sw ra 0(sp)
    sw s0 4(sp)
    sw s1 8(sp)
    sw s2 12(sp)
    sw s3 16(sp)

    mv s0 a0                # next element to print
    mv s1 a1                # rows still to print
    mv s2 a2                # values per row

pm_row:
    beqz s1 pm_done
    mv s3 s2                # values left in this row
pm_value:
    beqz s3 pm_newline
    lw a0 0(s0)
    jal put_int
    addi s0 s0 4
    addi s3 s3 -1
    beqz s3 pm_value        # no separator after the last value
    li a0 ' '
    jal put_char
    j pm_value
pm_newline:
    li a0 '\n'
    jal put_char
    addi s1 s1 -1
    j pm_row

pm_done:
    lw ra 0(sp)
    lw s0 4(sp)
    lw s1 8(sp)
    lw s2 12(sp)
    lw s3 16(sp)
    addi sp sp 20
    ret
