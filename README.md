# MLP in RISC-V Assembly

A small neural network that classifies handwritten digits (MNIST), with the whole forward pass written in RISC-V assembly: file I/O, matrix products, ReLU and argmax. The programs run in the Venus RISC-V simulator. The network is trained in NumPy with `tools/train.py` and exported as integer matrices that the assembly reads directly.

## The network

```
H = relu(W1 · X)        W1: 128 x 784
S = W2 · H              W2: 10 x 128
prediction = argmax(S)
```

- `X` holds one image per column: 784 raw pixel values from 0 to 255.
- There are no biases and no floating point. Every product and sum is a 32-bit integer operation.
- `W1` holds integers in [-127, 90] and `W2` holds integers in [-31, 24].

### Accuracy

| | Accuracy |
| --- | --- |
| Float model after training (MNIST test set, 10,000 images) | 97.93% |
| Integer model, as the assembly computes it (same 10,000 images) | 97.91% |
| `src/batch.s` in Venus on the 32 exported test digits | 32/32 = 100.0% |

The integer figure comes from a NumPy reference of the integer network. Running all 10,000 test images through the simulator would take hours, so only the 32-digit batch runs in Venus.

## Running

You need Java and a jar of the [Venus RISC-V simulator](https://github.com/ThaumicMekanism/venus). Keep the jar outside the repository, and run all commands from the repository root. The flag `-ms -1` lifts the simulator's default step limit, which a full forward pass exceeds.

### Classify one image

```sh
java -jar /path/to/venus.jar -ms -1 src/classify.s \
    data/model/w1.bin data/model/w2.bin data/digits/00.bin scores.bin
```

This prints `9`, the label of `00.bin`, and writes the scores to `scores.bin`. To read them:

```sh
python3 tools/matrix.py to-txt scores.bin
```

### Classify a batch

```
src/batch.s W1 W2 DIGITS LABELS SCORES
```

```sh
java -jar /path/to/venus.jar -ms -1 src/batch.s \
    data/model/w1.bin data/model/w2.bin \
    data/digits/digits.bin data/digits/labels.bin scores.bin
```

```
predictions: 9 8 6 0 8 1 2 3 7 0 1 3 8 1 9 1 3 7 6 7 2 5 0 4 4 5 0 2 6 9 5 4
labels:      9 8 6 0 8 1 2 3 7 0 1 3 8 1 9 1 3 7 6 7 2 5 0 4 4 5 0 2 6 9 5 4
accuracy:    32/32 = 100.0%
```

The images are the columns of one matrix, so the two matrix products process the whole batch at once. The accuracy is truncated to one decimal.

Exit codes for both programs:

| Code | Meaning |
| --- | --- |
| 1 | A file could not be opened, read, written or closed. |
| 2 | Matrix shapes do not fit together. |
| 3 | Wrong number of arguments. |
| 4 | The label count does not match the number of images (`batch.s` only). |

### Your own digit

```sh
python3 tools/matrix.py png my_digit.png my_digit.bin
python3 tools/matrix.py show my_digit.bin
java -jar /path/to/venus.jar -ms -1 src/classify.s \
    data/model/w1.bin data/model/w2.bin my_digit.bin scores.bin
```

The `png` command converts the image to grayscale and inverts it if the background is light. It then crops the image to the ink, scales it into a 20x20 box and centres it on a 28x28 canvas, the same framing as MNIST. `show` prints the result as text art.
