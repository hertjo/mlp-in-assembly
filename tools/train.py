#!/usr/bin/env python3
"""Train the 784-128-10 digit classifier and export it for the assembly code.

The network has no biases and a single ReLU hidden layer:

    scores = W2 @ relu(W1 @ x)

It is trained in floating point with NumPy (softmax cross-entropy, Adam,
mini-batches) on pixels scaled to [0, 1]. Afterwards both weight matrices are
rounded to small signed integers, and the model is evaluated exactly the way
the assembly runs it: raw pixels 0..255, 32-bit integer products and sums,
ReLU, argmax. Because ReLU commutes with a positive scale factor, the integer
network ranks the classes almost exactly like the float one.

The scales are picked so that no image can overflow a 32-bit accumulator:
for every output, the worst case over all inputs with pixels in 0..255 is
checked analytically, not only on the test set.

MNIST is read from a directory of the four original .gz files (IDX format).
Use --download to fetch them; keep that directory outside the repository.

Outputs (all in the binary matrix format of src/read_matrix.s):
    data/model/w1.bin        128 x 784 hidden-layer weights
    data/model/w2.bin        10 x 128 output-layer weights
    data/digits/digits.bin   784 x N, one test image per column
    data/digits/labels.bin   1 x N, true digit of every column
    data/digits/NN.bin       784 x 1, the same images one by one
    data/digits/labels.txt   the labels in plain text

Usage:
    python3 tools/train.py --mnist ~/datasets/mnist [--download] [--epochs 12]
"""

import argparse
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matrix  # noqa: E402

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
HIDDEN = 128
INT32_MAX = 2**31 - 1


# ------------------------------------------------------------------ data ----
def download(folder):
    os.makedirs(folder, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(MIRROR + name, path)


def read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = struct.unpack(">I", raw[:4])[0]
    ndim = magic & 0xFF
    if magic >> 8 != 0x08:  # 0x08 = unsigned byte
        raise SystemExit("%s: not an unsigned-byte IDX file" % path)
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(shape)


def load_mnist(folder):
    d = {k: read_idx(os.path.join(folder, v)) for k, v in FILES.items()}
    d["train_x"] = d["train_x"].reshape(-1, 784)
    d["test_x"] = d["test_x"].reshape(-1, 784)
    return d


# -------------------------------------------------------------- training ----
def train(x, y, epochs, batch, lr, seed):
    """Float training. x is (n, 784) in [0, 1], y is (n,) int."""
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((HIDDEN, 784)) * np.sqrt(2.0 / 784)).astype(np.float32)
    w2 = (rng.standard_normal((10, HIDDEN)) * np.sqrt(2.0 / HIDDEN)).astype(np.float32)
    params = [w1, w2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0

    n = len(x)
    for epoch in range(epochs):
        order = rng.permutation(n)
        rate = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))  # cosine decay
        loss_sum = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, yb = x[idx], y[idx]
            # forward (row-vector form: h = x W1^T)
            pre = xb @ w1.T
            h = np.maximum(pre, 0)
            s = h @ w2.T
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            loss_sum += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
            # backward
            g = p
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            gw2 = g.T @ h
            gh = (g @ w2) * (pre > 0)
            gw1 = gh.T @ xb
            # Adam
            step += 1
            for i, gr in enumerate((gw1, gw2)):
                m[i] = b1 * m[i] + (1 - b1) * gr
                v[i] = b2 * v[i] + (1 - b2) * gr * gr
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                params[i] -= (rate * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        print("epoch %2d  lr %.5f  loss %.4f" % (epoch + 1, rate, loss_sum / n))
    return w1, w2


def float_accuracy(w1, w2, x, y):
    s = np.maximum(x @ w1.T, 0) @ w2.T
    return float((s.argmax(axis=1) == y).mean())


# ---------------------------------------------------------- quantization ----
def worst_case_scores(q1, q2):
    """Largest |score| any 0..255 image can produce, per output, in int64."""
    hidden_max = 255 * np.clip(q1, 0, None).astype(np.int64).sum(axis=1)  # h_j <= this
    return np.abs(q2.astype(np.int64)) @ hidden_max


def quantize(w1, w2, bits1, bits2):
    """Round both layers to signed integers that cannot overflow int32."""
    q1 = np.round(w1 / np.abs(w1).max() * (2 ** (bits1 - 1) - 1)).astype(np.int32)
    for b in range(bits2, 1, -1):
        q2 = np.round(w2 / np.abs(w2).max() * (2 ** (b - 1) - 1)).astype(np.int32)
        if worst_case_scores(q1, q2).max() <= INT32_MAX:
            return q1, q2, b
    raise SystemExit("cannot fit the output layer into int32")


def int_forward(q1, q2, x):
    """Integer reference of the assembly. x is (784, n) int. Returns hidden, scores (int32)."""
    h = q1.astype(np.int64) @ x.astype(np.int64)
    h = np.maximum(h, 0)
    s = q2.astype(np.int64) @ h
    assert np.abs(h).max() <= INT32_MAX and np.abs(s).max() <= INT32_MAX
    return h.astype(np.int32), s.astype(np.int32)


# ---------------------------------------------------------------- export ----
def pick_digits(x, y, count, seed):
    """count test images, as balanced over the ten classes as possible."""
    rng = np.random.default_rng(seed)
    per_class = [rng.permutation(np.flatnonzero(y == c)) for c in range(10)]
    chosen = [per_class[i % 10][i // 10] for i in range(count)]
    chosen = rng.permutation(chosen)
    return chosen


def export(q1, q2, x, y, out_dir):
    model = os.path.join(out_dir, "model")
    digits = os.path.join(out_dir, "digits")
    os.makedirs(model, exist_ok=True)
    os.makedirs(digits, exist_ok=True)
    for name in os.listdir(digits):
        if name.endswith(".bin") or name.endswith(".txt"):
            os.remove(os.path.join(digits, name))

    matrix.write_bin(os.path.join(model, "w1.bin"), q1)
    matrix.write_bin(os.path.join(model, "w2.bin"), q2)

    cols = x.T.astype(np.int32)  # (784, n)
    matrix.write_bin(os.path.join(digits, "digits.bin"), cols)
    matrix.write_bin(os.path.join(digits, "labels.bin"), y.reshape(1, -1).astype(np.int32))
    for i in range(len(y)):
        matrix.write_bin(os.path.join(digits, "%02d.bin" % i), cols[:, i:i + 1])
    with open(os.path.join(digits, "labels.txt"), "w") as f:
        f.write("".join("%02d %d\n" % (i, d) for i, d in enumerate(y)))


def main():
    ap = argparse.ArgumentParser(description="Train and export the integer MLP.")
    ap.add_argument("--mnist", required=True, help="folder with the four MNIST .gz files")
    ap.add_argument("--download", action="store_true", help="fetch missing files first")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bits1", type=int, default=8, help="bits for w1 (sign included)")
    ap.add_argument("--bits2", type=int, default=8, help="max bits for w2 (sign included)")
    ap.add_argument("--digits", type=int, default=32, help="test images to export")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    if args.download:
        download(args.mnist)
    d = load_mnist(args.mnist)
    xtr = d["train_x"].astype(np.float32) / 255
    xte = d["test_x"].astype(np.float32) / 255

    w1, w2 = train(xtr, d["train_y"].astype(np.int64), args.epochs, args.batch, args.lr, args.seed)
    print("float test accuracy:   %.2f%%" % (100 * float_accuracy(w1, w2, xte, d["test_y"])))

    q1, q2, bits2 = quantize(w1, w2, args.bits1, args.bits2)
    _, scores = int_forward(q1, q2, d["test_x"].T)
    acc = float((scores.argmax(axis=0) == d["test_y"]).mean())
    print("w1 in [%d, %d], w2 in [%d, %d] (%d bits), worst-case |score| %d"
          % (q1.min(), q1.max(), q2.min(), q2.max(), bits2, worst_case_scores(q1, q2).max()))
    print("integer test accuracy: %.2f%% (10000 images)" % (100 * acc))

    pick = pick_digits(d["test_x"], d["test_y"], args.digits, args.seed)
    export(q1, q2, d["test_x"][pick], d["test_y"][pick], args.out)
    print("exported model and %d digits to %s" % (len(pick), os.path.relpath(args.out, os.getcwd())))


if __name__ == "__main__":
    main()
