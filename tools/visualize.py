#!/usr/bin/env python3
"""Render assets/demo.gif from a real Venus run of batch.s.

The script runs src/batch.s in the Venus RISC-V simulator on the test digits
in data/digits, with the optional hidden-layer output, and draws one scene per
image: the 28x28 input, the ReLU activations of the hidden layer, the 10
output scores and the predicted digit. Every number in the GIF comes from the
matrices that the assembly program wrote; numpy is only used to read them.

Requirements: Java, the Venus jar, numpy and Pillow.

Usage:
    python3 tools/visualize.py --venus /path/to/venus.jar [--java java]
                               [--count 12] [--out assets/demo.gif]

The jar path can also be given through the VENUS_JAR environment variable.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from matrix import read_bin as read_matrix  # noqa: E402

# ---------------------------------------------------------------- style ----
W, H = 960, 400
BG = (13, 17, 23)
PANEL = (22, 27, 34)
GRID = (36, 43, 52)
TEXT = (230, 237, 243)
MUTED = (125, 133, 144)
ACCENT = (88, 166, 255)
GOOD = (63, 185, 80)
BAD = (248, 81, 73)
# heatmap: background -> deep blue -> blue -> cyan -> near white
HEAT = [(22, 27, 34), (25, 55, 110), (45, 110, 210), (90, 190, 240), (225, 245, 255)]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
]

BUILD_FRAMES = 14   # frames used to animate one forward pass
HOLD_FRAMES = 10    # frames the finished result stays on screen
FRAME_MS = 70
HOLD_MS = 140


def font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def heat(v):
    """Map v in [0, 1] onto the heatmap gradient."""
    v = min(max(v, 0.0), 1.0) * (len(HEAT) - 1)
    i = min(int(v), len(HEAT) - 2)
    f = v - i
    a, b = HEAT[i], HEAT[i + 1]
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


# ------------------------------------------------------------ venus run ----
def run_venus(java, jar, workdir):
    model = os.path.join(ROOT, "data", "model")
    digits = os.path.join(ROOT, "data", "digits")
    paths = {
        "inputs": os.path.join(digits, "digits.bin"),
        "labels": os.path.join(digits, "labels.bin"),
        "scores": os.path.join(workdir, "scores.bin"),
        "hidden": os.path.join(workdir, "hidden.bin"),
    }
    cmd = [java, "-jar", jar, "-ms", "-1", "src/batch.s",
           os.path.join(model, "w1.bin"), os.path.join(model, "w2.bin"),
           paths["inputs"], paths["labels"], paths["scores"], paths["hidden"]]
    print("running Venus on data/digits/digits.bin ...")
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("Venus failed (%d):\n%s%s" % (out.returncode, out.stdout, out.stderr))
    print(out.stdout.strip())

    preds = [int(x) for x in re.search(r"predictions:\s*([-\d ]+)", out.stdout).group(1).split()]
    accuracy = re.search(r"accuracy:\s*(.+)", out.stdout).group(1).strip()
    return {
        "inputs": read_matrix(paths["inputs"]),
        "labels": read_matrix(paths["labels"])[0],
        "scores": read_matrix(paths["scores"]),
        "hidden": read_matrix(paths["hidden"]),
        "preds": preds,
        "accuracy": accuracy,
    }


# -------------------------------------------------------------- drawing ----
def centered(draw, xy, text, fnt, fill):
    x, y = xy
    w = draw.textlength(text, font=fnt)
    draw.text((x - w / 2, y), text, font=fnt, fill=fill)


def draw_frame(run, idx, t, fonts):
    f_title, f_label, f_small, f_big = fonts
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    n = run["shown"]
    label = int(run["labels"][idx])
    pred = run["preds"][idx]
    hidden = run["hidden"][:, idx].astype(float)
    scores = run["scores"][:, idx].astype(float)
    pixels = run["inputs"][:, idx].reshape(28, 28).astype(float)

    # header
    d.text((32, 22), "MLP in RISC-V assembly", font=f_title, fill=TEXT)
    d.text((32, 50), "input 784  \u2192  W1 \u00b7 x, ReLU  \u2192  hidden %d  \u2192  W2 \u00b7 h  \u2192  scores 10  \u2192  argmax"
           % hidden.size, font=f_small, fill=MUTED)
    tag = "image %d / %d" % (idx + 1, n)
    d.text((W - 32 - d.textlength(tag, font=f_small), 28), tag, font=f_small, fill=MUTED)

    top = 100
    # ---- input digit
    cell = 8
    x0, y0 = 32, top + 20
    d.text((x0, top - 4), "input 28x28", font=f_label, fill=MUTED)
    d.rounded_rectangle((x0 - 6, y0 - 6, x0 + 28 * cell + 6, y0 + 28 * cell + 6), 8, fill=PANEL)
    peak = max(pixels.max(), 1.0)
    for r in range(28):
        for c in range(28):
            v = pixels[r, c] / peak
            if v > 0:
                g = int(PANEL[0] + (TEXT[0] - PANEL[0]) * v)
                d.rectangle((x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 1,
                             y0 + (r + 1) * cell - 1), fill=(g, g + 3 if g < 252 else g, g + 6 if g < 249 else g))

    # ---- hidden layer strip: 128 cells as 4 rows of 32
    hx, hy = 300, top + 20
    rows, cols = 4, hidden.size // 4
    hc = 11
    gap = 2
    d.text((hx, top - 4), "hidden ReLU activations", font=f_label, fill=MUTED)
    active = int((hidden > 0).sum())
    hmax = max(run["hidden"].max(), 1)
    reveal = ease(t / 0.45)
    shown = int(round(reveal * hidden.size))
    for k in range(hidden.size):
        r, c = divmod(k, cols)
        v = np.sqrt(hidden[k] / hmax) if k < shown else 0.0
        color = heat(v) if v > 0 else GRID
        px, py = hx + c * (hc + gap), hy + r * (hc + gap)
        d.rectangle((px, py, px + hc - 1, py + hc - 1), fill=color)
    strip_w = cols * (hc + gap) - gap
    d.text((hx, hy + rows * (hc + gap) + 8), "%d of %d units active" % (active if shown == hidden.size else
           int((hidden[:shown] > 0).sum()), hidden.size), font=f_small, fill=MUTED)
    # colour scale, right-aligned on the same line as the unit count
    sy = hy + rows * (hc + gap) + 13
    sw = 100
    zero_w = d.textlength("0 ", font=f_small)
    sx = hx + strip_w - d.textlength(" max", font=f_small) - sw
    d.text((sx - zero_w, sy - 5), "0", font=f_small, fill=MUTED)
    for i in range(sw):
        d.line((sx + i, sy, sx + i, sy + 6), fill=heat(i / (sw - 1)))
    d.text((sx + sw + d.textlength(" ", font=f_small), sy - 5), "max", font=f_small, fill=MUTED)

    # ---- output scores
    bx, by = hx, top + 150
    bgap, bh = 10, 92
    bw = (strip_w - 9 * bgap) // 10
    grow = ease((t - 0.35) / 0.45)
    smax = max(np.abs(scores).max(), 1.0)
    base = by + bh
    d.text((bx, by - 22), "output scores", font=f_label, fill=MUTED)
    for k in range(10):
        px = bx + k * (bw + bgap)
        d.rectangle((px, base - bh, px + bw, base), fill=PANEL)
        h = scores[k] / smax * bh * grow
        color = ACCENT if (k == pred and t >= 0.8) else (70, 80, 94)
        if h >= 0:
            d.rectangle((px, base - h, px + bw, base), fill=color)
        else:
            d.rectangle((px, base, px + bw, base - h * 0.25), fill=color)
        centered(d, (px + bw / 2, base + 10), str(k), f_small, TEXT if k == pred and t >= 0.8 else MUTED)

    # ---- prediction
    cx = W - 32 - 80
    d.text((cx - 70, top - 4), "prediction", font=f_label, fill=MUTED)
    d.rounded_rectangle((cx - 80, top + 14, cx + 80, top + 254), 12, fill=PANEL)
    if t >= 0.8:
        a = ease((t - 0.8) / 0.2)
        col = tuple(int(PANEL[i] + (TEXT[i] - PANEL[i]) * a) for i in range(3))
        centered(d, (cx, top + 40), str(pred), f_big, col)
        ok = pred == label
        verdict = ("correct" if ok else "wrong, label %d" % label)
        vcol = GOOD if ok else BAD
        vcol = tuple(int(PANEL[i] + (vcol[i] - PANEL[i]) * a) for i in range(3))
        centered(d, (cx, top + 210), verdict, f_label, vcol)

    # footer
    foot = "batch.s accuracy over all %d test digits: %s" % (len(run["preds"]), run["accuracy"])
    d.text((32, H - 34), foot, font=f_small, fill=MUTED)
    regen = "tools/visualize.py"
    d.text((W - 32 - d.textlength(regen, font=f_small), H - 34), regen, font=f_small, fill=MUTED)
    return img


def render(run, out_path, count):
    run["shown"] = min(count, len(run["preds"]))
    fonts = (font(20), font(14), font(12), font(150))
    frames, durations = [], []
    for idx in range(run["shown"]):
        for i in range(BUILD_FRAMES):
            frames.append(draw_frame(run, idx, i / (BUILD_FRAMES - 1), fonts))
            durations.append(FRAME_MS)
        frames.append(draw_frame(run, idx, 1.0, fonts))
        durations.append(HOLD_MS * HOLD_FRAMES)

    # One shared palette, built from the finished frame of every image, keeps
    # the colours stable between frames and the file small.
    holds = [f for f, ms in zip(frames, durations) if ms > FRAME_MS]
    palette_src = Image.new("RGB", (W, H * len(holds)))
    for i, f in enumerate(holds):
        palette_src.paste(f, (0, H * i))
    pal = palette_src.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    quant = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    quant[0].save(out_path, save_all=True, append_images=quant[1:], duration=durations,
                  loop=0, optimize=True, disposal=1)
    print("wrote %s (%d frames, %.2f MB)" % (out_path, len(quant), os.path.getsize(out_path) / 1e6))


def main():
    parser = argparse.ArgumentParser(description="Render the README demo GIF from a Venus run.")
    parser.add_argument("--venus", default=os.environ.get("VENUS_JAR"), help="path to the Venus jar")
    parser.add_argument("--java", default=os.environ.get("JAVA", "java"), help="java executable")
    parser.add_argument("--count", type=int, default=12, help="images to animate")
    parser.add_argument("--out", default=os.path.join(ROOT, "assets", "demo.gif"))
    args = parser.parse_args()
    if not args.venus:
        parser.error("pass --venus /path/to/venus.jar or set VENUS_JAR")

    with tempfile.TemporaryDirectory() as tmp:
        run = run_venus(args.java, os.path.abspath(args.venus), tmp)
    render(run, os.path.abspath(args.out), args.count)


if __name__ == "__main__":
    main()
