# Data Compression Project
## Experiment 1: LLM-Guided Arithmetic Coding on Enwik8

**Slogan:** Push the Boundaries with AI!

---

### The Core Idea

Arithmetic coding is theoretically optimal — it can approach the true entropy
H(X) of a source — but only if it has an accurate probability model.
Classical compressors use simple n-gram statistics, which miss long-range
linguistic patterns.

**This project replaces the statistical model with GPT-2**, a large language
model that has learned rich probability distributions over text.  At each
token position, GPT-2 predicts P(next\_token | all previous tokens); these
probabilities are fed directly into a 32-bit integer arithmetic coder.

This is the method described formally in:
> Delétang et al., *Language Modeling Is Compression*, Google DeepMind, 2023.
> https://arxiv.org/abs/2309.10668

---

### File Structure

```
lm_arithmetic_coding/
├── arithmetic_coder.py   # Integer arithmetic encoder/decoder (32-bit, E1/E2/E3 rescaling)
├── baselines.py          # Shannon entropy, Huffman, static arithmetic coding, gzip, bzip2, lzma
├── lm_compressor.py      # GPT-2 + arithmetic coding with KV-cache acceleration
├── evaluate.py           # Main evaluation runner (single model, both conditions)
├── ablation_window.py    # Window-size ablation (W ∈ {128, 256, 512}) under both conditions
├── test_edge_cases.py    # Losslessness and CDF correctness verification
├── download_data.py      # Download Enwik8 dataset
├── project_runs.ipynb    # Jupyter notebook — end-to-end experiment walkthrough
├── requirements.txt      # Python dependencies
├── data/                 # Created automatically on first run
│   └── enwik8_1000k.txt
└── results/              # Timestamped JSON + report files (created on first run)
```

---

### Setup

```bash
pip install -r requirements.txt
```

If you have a GPU, torch will use it automatically. CPU-only is fine for
the cross-entropy evaluation; the full encode/decode is much faster with a GPU.

---

### Running the Experiments

**Quick run — cross-entropy BPC only (Condition A, ~5 minutes on GPU):**
```bash
python evaluate.py
```

**Condition B (offset 300 KB, prose-dominant segment):**
```bash
python evaluate.py --offset 300000
```

**With actual encode + decode verification:**
```bash
python evaluate.py --full-encode
```

**Larger GPT-2 variants (better compression, slower):**
```bash
python evaluate.py --model gpt2-medium
python evaluate.py --model gpt2-xl
```

**Window-size ablation under both conditions:**
```bash
python ablation_window.py
python ablation_window.py --offset 300000
```

**Edge case and losslessness verification:**
```bash
python test_edge_cases.py
```

---

### Results

Measured on 50 KB of Enwik8, W = 512.

| Method                  | Cond. A BPC | Cond. A Ratio | Cond. B BPC | Cond. B Ratio |
|-------------------------|:-----------:|:-------------:|:-----------:|:-------------:|
| Static Huffman          | 4.919       | 1.63×         | 5.004       | 1.60×         |
| gzip -9                 | 3.013       | 2.66×         | 3.128       | 2.56×         |
| bzip2 -9                | 2.696       | 2.97×         | 2.787       | 2.87×         |
| LLM-AC GPT-2            | 1.118       | 7.16×         | 1.181       | 6.77×         |
| LLM-AC GPT-2-medium     | 0.986       | 8.11×         | 1.039       | 7.70×         |
| **LLM-AC GPT-2-XL**     | **0.874**   | **9.15×**     | **0.958**   | **8.35×**     |
| PAQ8 (state-of-the-art) | ~1.0        | ~8×           | —           | —             |

**Condition A** (offset 0): opening segment of the Wikipedia XML dump — ~53% XML markup lines, ~47% prose.  Directly comparable to the Hutter Prize leaderboard.  
**Condition B** (offset 300 KB): prose-dominant passage (Abraham Lincoln / Aristotle articles) — ~6% XML, ~94% prose.

---

### How It Works (step by step)

1. **Tokenize** the text using GPT-2's BPE tokenizer.
2. For each token position `i`:
   - Incrementally run a forward pass using KV-cache for O(n) amortised cost.
   - Read the softmax probabilities over all 50,257 vocabulary tokens.
   - Convert to an integer CDF (scaled to 10,000,000 counts for precision).
   - Feed the token's CDF range to the arithmetic encoder.
3. The encoder outputs a bitstream. Total bits ≈ Σ –log₂ P(token_i | context).
4. Decoding is symmetric: the decoder queries GPT-2 at each step using
   already-decoded tokens as context, so no side information is needed.

---

### AI Integration

This project uses AI at **the algorithm level**, not just for coding help:

- GPT-2 *is* the probability model — it replaces the entire statistical
  component of the arithmetic coder.
- Claude was used to design the E1/E2/E3 rescaling strategy for the integer
  arithmetic coder and to debug the CDF scaling precision issue.
- Interaction history and bug fixes are documented in `../ai_log.docx`.

---

### Evaluation Metrics

- **Bits per character (BPC):** primary metric. Lower is better.
- **Compression ratio:** 8 ÷ BPC.
- **Lossless verification:** decoded output must exactly match the original.
- **Encoding speed:** tokens/second.
