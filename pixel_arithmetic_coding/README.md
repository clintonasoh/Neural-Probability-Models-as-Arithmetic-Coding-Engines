# Data Compression Project
## Experiment 2: PixelCNN-Guided Arithmetic Coding on Fashion-MNIST

---

### The Core Idea

Arithmetic coding is only as good as its probability model.  Classical image
compressors use hand-crafted predictors (DCT in JPEG, LZ in PNG) that miss
spatial structure learned from data.

**This project replaces the statistical model with a PixelCNN**, a convolutional
neural network that models the joint pixel distribution autoregressively.  At each
pixel position $(i, j)$, PixelCNN predicts $P(\text{pixel}_{i,j} \mid \text{all
prior pixels})$ via a 256-class softmax; these probabilities are fed directly into
a 32-bit integer arithmetic coder.

This is an image-domain instance of the framework described in:
> van den Oord et al., *Pixel Recurrent Neural Networks*, ICML, 2016.
> https://arxiv.org/abs/1601.06759

---

### File Structure

```
pixel_arithmetic_coding/
├── pixelcnn.py          # PixelCNN model (masked convolutions, residual blocks)
├── train.py             # Training script — Fashion-MNIST, 20 epochs
├── pixel_compressor.py  # Pixel-level encode/decode using PixelCNN + arithmetic coder
├── arithmetic_coder.py  # 32-bit integer arithmetic encoder/decoder (E1/E2/E3)
├── baselines.py         # Raw, Shannon entropy, gzip, PNG baselines
├── evaluate.py          # Main evaluation runner
├── download_data.py     # Download and cache Fashion-MNIST
├── test_edge_cases.py   # Edge case and losslessness verification
├── pixel_runs.ipynb     # Jupyter notebook — end-to-end experiment walkthrough
├── requirements.txt     # Python dependencies
├── checkpoints/         # Saved model checkpoints (created by train.py)
│   └── pixelcnn_fmnist.pt
└── results/             # Timestamped JSON + report files (created on first run)
    ├── training_log.json
    ├── training_curve.png
    ├── pixelcnn_fmnist_*.json
    └── edge_case_report.txt
```

---

### Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training.  Inference (evaluation
and encode/decode) is feasible on CPU, but each image requires 784 sequential
forward passes (~2.4 s encode, ~2.5 s decode on a Tesla T4 GPU).

---

### Running the Experiments

**Step 1 — Train the PixelCNN (~10 minutes on a Tesla T4):**
```bash
python train.py
```
Checkpoint saved to `checkpoints/pixelcnn_fmnist.pt`.  Training and test BPP are
logged per epoch to `results/training_log.json`.

**Step 2 — Theoretical BPP on 500 test images:**
```bash
python evaluate.py --n-images 500
```

**Step 3 — Full encode/decode verification (5 images):**
```bash
python evaluate.py --full-encode --n-images 100
```

**Step 4 — Edge case verification:**
```bash
python test_edge_cases.py
```

---

### Results

| Method                   | BPP       | Ratio     | Notes                          |
|--------------------------|----------:|:---------:|--------------------------------|
| Raw (8 bpp)              | 8.000     | 1.00×     | Uncompressed baseline          |
| PNG (lossless)           | 5.190     | 1.54×     | 500 test images                |
| gzip -9 (raw pixels)     | 4.814     | 1.66×     | 500 test images                |
| Shannon entropy          | 4.162     | 1.92×     | Theoretical lower bound        |
| PixelCNN-AC (theoretical)| 2.968     | 2.70×     | 500 test images, no AC overhead|
| **PixelCNN-AC (actual)** | **2.630** | **3.04×** | 5 images, 100% lossless        |

Training converged from 3.892 BPP (epoch 1) to 2.957 BPP (epoch 20) with no
train/test gap, indicating a well-matched model capacity.

---

### How It Works

1. **Train** PixelCNN to minimise pixel-level cross-entropy on Fashion-MNIST.
2. **Encoding** — for each of 784 pixels in raster order:
   - Pass the partial image (pixels revealed so far) through PixelCNN.
   - Read the 256-class softmax output for the current pixel position.
   - Scale to an integer CDF summing to 10,000,000.
   - Feed the true pixel's CDF range to the arithmetic encoder.
3. **Decoding** is symmetric: the decoder queries PixelCNN with the same growing
   context buffer, recovers each pixel from the arithmetic decoder, and appends
   it to the buffer.  Since both sides use identical model state at every step,
   lossless reconstruction is guaranteed.

> **Critical design note:** both encoder and decoder run 784 sequential forward
> passes (one per pixel) rather than a single pass over the full image.  This
> ensures CDF symmetry at every step — a single-pass encoder violates this
> invariant due to floating-point non-commutativity across the residual blocks,
> causing decode failures.

---

### PixelCNN Architecture

| Component            | Details                                   |
|----------------------|-------------------------------------------|
| Input embedding      | Type-A MaskConv 7×7 (no self-connection)  |
| Residual blocks      | 8 × (Type-B MaskConv 3×3 + BN + ReLU)    |
| Output head          | 1×1 conv → 256 logits per pixel           |
| Feature channels     | 64                                        |
| Parameters           | ~1.2 M                                    |
| Optimiser            | Adam, lr = 1e-3, batch = 128              |
| Training             | 20 epochs on Fashion-MNIST train set      |

