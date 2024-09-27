#!/usr/bin/env python3
"""Convert matrices between the formats used in this repository.

Binary format (what the assembly reads and writes):
    int32 rows, int32 cols, then rows*cols int32 values in row-major order,
    everything little-endian.

Text format:
    first line "rows cols", then one line per row with the values separated
    by spaces.

Commands:
    to-txt  IN.bin [OUT.txt]       binary -> text (stdout if OUT is omitted)
    to-bin  IN.txt OUT.bin         text -> binary
"""

import argparse
import sys

import numpy as np


def read_bin(path):
    raw = np.fromfile(path, dtype="<i4")
    if raw.size < 2:
        raise SystemExit("%s: too short for a matrix header" % path)
    rows, cols = int(raw[0]), int(raw[1])
    if raw.size != 2 + rows * cols:
        raise SystemExit("%s: header says %dx%d but holds %d values" % (path, rows, cols, raw.size - 2))
    return raw[2:].reshape(rows, cols).astype(np.int32)


def write_bin(path, m):
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    header = np.array(m.shape, dtype="<i4")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(m, dtype="<i4").tobytes())


def read_txt(path):
    with open(path) as f:
        tokens = f.read().split()
    rows, cols = int(tokens[0]), int(tokens[1])
    values = [int(t) for t in tokens[2:]]
    if len(values) != rows * cols:
        raise SystemExit("%s: header says %dx%d but holds %d values" % (path, rows, cols, len(values)))
    return np.array(values, dtype=np.int32).reshape(rows, cols)


def format_txt(m):
    lines = ["%d %d" % m.shape]
    lines += [" ".join(str(int(v)) for v in row) for row in m]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Matrix format conversions.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-txt")
    p.add_argument("src")
    p.add_argument("dst", nargs="?")
    p = sub.add_parser("to-bin")
    p.add_argument("src")
    p.add_argument("dst")
    args = ap.parse_args()

    if args.cmd == "to-txt":
        text = format_txt(read_bin(args.src))
        if args.dst:
            with open(args.dst, "w") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
    elif args.cmd == "to-bin":
        write_bin(args.dst, read_txt(args.src))


if __name__ == "__main__":
    main()
