# sys.s
# Environment calls of the Venus RISC-V simulator, wrapped so that they follow
# the normal calling convention (arguments from a0 on, result in a0). The
# simulator wants the call number in a0 and the arguments from a1 on, so each
# wrapper moves its arguments up by one register before the ecall.

.globl put_int, put_char, put_str, alloc, quit, fail

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
