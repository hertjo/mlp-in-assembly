#!/usr/bin/env python3
"""Run every assembly program in Venus and compare against expectations.

1. Unit tests: each tests/test_*.s lists its expected output in lines that
   start with "#> " and, if it should fail, its exit code in a "#! exit N"
   line. The program's output must match exactly.
2. batch.s on data/digits: the scores and hidden activations that Venus
   writes must equal the NumPy integer reference bit for bit, and the printed
   predictions must equal the reference argmax.
3. classify.s on every single digit (skip with --no-singles): same check
   for the 10 scores and the printed digit.

Usage:
    python3 tools/check.py --venus /path/to/venus.jar [--java java] [--no-singles]
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matrix  # noqa: E402

W1 = os.path.join(ROOT, "data", "model", "w1.bin")
W2 = os.path.join(ROOT, "data", "model", "w2.bin")
DIGITS = os.path.join(ROOT, "data", "digits")


def reference(q1, q2, x):
    """The network as the assembly computes it, with 32-bit wrap-around."""
    h = (q1.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)
    h = np.maximum(h, 0)
    s = (q2.astype(np.int64) @ h.astype(np.int64)).astype(np.int32)
    return h, s


def venus(java, jar, *args):
    cmd = [java, "-jar", jar, "-ms", "-1"] + list(args)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def run_unit_tests(java, jar):
    failures = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.s"))):
        with open(path) as f:
            lines = f.read().splitlines()
        expected = [l[3:] for l in lines if l.startswith("#> ")]
        code = 0
        for l in lines:
            m = re.match(r"#! exit (\d+)", l)
            if m:
                code = int(m.group(1))
        r = venus(java, jar, os.path.relpath(path, ROOT))
        got = r.stdout.rstrip("\n").splitlines()
        ok = r.returncode == code and (code != 0 or got == expected)
        failures += not ok
        print("  %-28s %s" % (os.path.basename(path), "ok" if ok else "FAILED"))
        if not ok:
            print("    exit %d (want %d)\n    got:  %r\n    want: %r" % (r.returncode, code, got, expected))
    return failures


def run_batch(java, jar, q1, q2, tmp):
    x = matrix.read_bin(os.path.join(DIGITS, "digits.bin"))
    labels = matrix.read_bin(os.path.join(DIGITS, "labels.bin")).ravel()
    h_ref, s_ref = reference(q1, q2, x)
    scores, hidden = os.path.join(tmp, "scores.bin"), os.path.join(tmp, "hidden.bin")
    r = venus(java, jar, "src/batch.s", W1, W2, os.path.join(DIGITS, "digits.bin"),
              os.path.join(DIGITS, "labels.bin"), scores, hidden)
    print("\n".join("  " + l for l in r.stdout.strip().splitlines()))
    if r.returncode != 0:
        print("  batch.s exited with %d" % r.returncode)
        return 1
    preds = [int(v) for v in re.search(r"predictions:([-\d ]+)", r.stdout).group(1).split()]
    checks = {
        "hidden bit for bit": np.array_equal(matrix.read_bin(hidden), h_ref),
        "scores bit for bit": np.array_equal(matrix.read_bin(scores), s_ref),
        "predictions": preds == list(s_ref.argmax(axis=0)),
    }
    for name, ok in checks.items():
        print("  %-28s %s" % (name, "ok" if ok else "FAILED"))
    print("  reference accuracy           %d/%d" % ((s_ref.argmax(axis=0) == labels).sum(), len(labels)))
    return sum(not ok for ok in checks.values())


def run_singles(java, jar, q1, q2, tmp):
    failures = 0
    paths = sorted(glob.glob(os.path.join(DIGITS, "[0-9][0-9].bin")))
    out = os.path.join(tmp, "out.bin")
    for path in paths:
        _, s_ref = reference(q1, q2, matrix.read_bin(path))
        r = venus(java, jar, "src/classify.s", W1, W2, path, out)
        ok = (r.returncode == 0 and r.stdout.strip() == str(int(s_ref.argmax()))
              and np.array_equal(matrix.read_bin(out), s_ref))
        failures += not ok
        if not ok:
            print("  %s FAILED: %r" % (os.path.basename(path), r.stdout.strip()))
    print("  %d of %d single images match bit for bit" % (len(paths) - failures, len(paths)))
    return failures


def main():
    ap = argparse.ArgumentParser(description="Check the assembly against the NumPy reference.")
    ap.add_argument("--venus", default=os.environ.get("VENUS_JAR"), help="path to the Venus jar")
    ap.add_argument("--java", default=os.environ.get("JAVA", "java"))
    ap.add_argument("--no-singles", action="store_true", help="skip classify.s on each digit")
    args = ap.parse_args()
    if not args.venus:
        ap.error("pass --venus /path/to/venus.jar or set VENUS_JAR")
    jar = os.path.abspath(args.venus)

    q1, q2 = matrix.read_bin(W1), matrix.read_bin(W2)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        print("unit tests")
        failures += run_unit_tests(args.java, jar)
        print("batch.s")
        failures += run_batch(args.java, jar, q1, q2, tmp)
        if not args.no_singles:
            print("classify.s")
            failures += run_singles(args.java, jar, q1, q2, tmp)
    print("all checks passed" if failures == 0 else "%d check(s) failed" % failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
