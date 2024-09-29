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
    png     IN.png OUT.bin         image -> 784 x 1 input column (0..255)
    show    IN.bin [--col N]       draw a 784-row input column as text art

The png command converts any image to grayscale, crops it to the bounding box
of the ink, scales it into a 20x20 box and centres it on a 28x28 canvas, the
same framing as the MNIST images. A light background is inverted
automatically so that ink is bright, as in MNIST; use --no-invert to keep the
image as it is.
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


def png_to_column(path, invert=None):
    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if invert is None:
        # border pixels tell us the background colour
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        invert = border.mean() > 127
    if invert:
        img = 255 - img
    img -= img.min()
    if img.max() > 0:
        img *= 255 / img.max()

    ys, xs = np.nonzero(img > 32)
    if len(ys):
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape
    scale = 20 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
                       dtype=np.float64)

    canvas = np.zeros((28, 28))
    # centre by the centre of mass, like the original MNIST preprocessing
    total = small.sum()
    if total > 0:
        cy = (small.sum(axis=1) @ np.arange(nh)) / total
        cx = (small.sum(axis=0) @ np.arange(nw)) / total
    else:
        cy, cx = nh / 2, nw / 2
    top = int(round(14 - cy))
    left = int(round(14 - cx))
    top = min(max(top, 0), 28 - nh)
    left = min(max(left, 0), 28 - nw)
    canvas[top:top + nh, left:left + nw] = small
    return np.clip(np.round(canvas), 0, 255).astype(np.int32).reshape(784, 1)


def show(m, col):
    pixels = m[:, col].reshape(28, 28)
    ramp = " .:-=+*#%@"
    peak = max(int(pixels.max()), 1)
    for row in pixels:
        print("".join(ramp[min(int(v) * len(ramp) // (peak + 1), len(ramp) - 1)] * 2 for v in row))


def main():
    ap = argparse.ArgumentParser(description="Matrix format conversions.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("to-txt")
    p.add_argument("src")
    p.add_argument("dst", nargs="?")
    p = sub.add_parser("to-bin")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("png")
    p.add_argument("src")
    p.add_argument("dst")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--invert", dest="invert", action="store_true", default=None)
    g.add_argument("--no-invert", dest="invert", action="store_false")
    p = sub.add_parser("show")
    p.add_argument("src")
    p.add_argument("--col", type=int, default=0)
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
    elif args.cmd == "png":
        write_bin(args.dst, png_to_column(args.src, args.invert))
    elif args.cmd == "show":
        m = read_bin(args.src)
        if m.shape[0] != 784:
            raise SystemExit("expected 784 rows, got %d" % m.shape[0])
        show(m, args.col)


if __name__ == "__main__":
    main()
