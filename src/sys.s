# sys.s
# Environment calls of the Venus RISC-V simulator, wrapped so that they follow
# the normal calling convention (arguments from a0 on, result in a0). The
# simulator wants the call number in a0 and the arguments from a1 on, so each
# wrapper moves its arguments up by one register before the ecall.

.globl put_int, put_char, put_str, alloc, quit, fail
.globl file_open, file_read, file_write, file_close

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
