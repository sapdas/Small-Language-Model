# Academic Small Language Model (128-Token Micro-GPT Inference Engine)

A transparent, **inference-only** Decoder-Only Transformer (GPT-style Small Language Model) with a **128-token vocabulary ($|V| = 128 > 100$)** implemented in pure Python and NumPy for academic and instructional demonstration.

---

## 1. Pedagogical Objectives & Key Features

1. **Strictly Inference-Only (No Training Loop)**:
   - Operates exclusively on a **known, deterministic set of 6,888 parameters** stored in [`parameters.json`](parameters.json) and analytically constructed in [`parameter_source.py`](parameter_source.py).
   - Zero backpropagation, zero optimizers, and zero opaque binary checkpoints.
2. **128-Token Vocabulary Across 8 Syntactic Categories & 5 Knowledge Domains**:
   - Covers `Control & Punctuation` (`4`), `Determiners` (`8`), `Adjectives` (`20`), `Subject Nouns` (`28`), `Verbs` (`24`), `Prepositions` (`12`), `Object Nouns` (`24`), and `Adverbs` (`8`) spanning **Physical/Nature**, **Academic/Scholarship**, **AI/Computing**, **Medicine/Biology**, and **Astronomy/Space**.
3. **Complete Parameter Source & Provenance (`d_model = 16`, `H = 4`, `d_ff = 40`)**:
   - Every weight matrix ($W_E, W_P, \gamma_1, \beta_1, W_Q, W_K, W_V, W_O, \gamma_2, \beta_2, W_1, b_1, W_2, b_2, \gamma_f, \beta_f, W_{\text{LM}}, b_{\text{LM}}$) is derived from an explicit **16-dimensional semantic-lexical subspace**, **3-bit hypercube lexical coding**, and **LayerNorm hypersphere matched-filter geometry** (with exact sphere radius $\sqrt{16} = 4.0000$).
4. **White-Box Inner Working Inspection & Causal Ablation**:
   - Captures all 8 mathematical stages of the Transformer forward pass inside a [`ForwardPassTrace`](slm_inference.py) structure and visualizes intermediate vectors, LayerNorm sphere projections, 4-Head Causal Self-Attention masks and heatmaps, 40 FFN Key-Value memory neuron activations, and 128-token unembedding logits.
5. **Autoregressive Next-Token Prediction & Token String Growth**:
   - Demonstrates step-by-step how the final position's 16D hidden state predicts a probability distribution over all 128 vocabulary tokens, selects the next token (via Greedy, Temperature, or Top-$k$ decoding), and **appends the predicted token to the running token sequence** until `<EOS>`.

---

## 2. Architecture & Mathematical Pipeline (`|V| = 128, d_model = 16`)

```
Input String: "<BOS> the cat sat on the"
      │
      ▼  [Stage 1: Tokenization & 128-Token Vocabulary Lookup]
Token IDs: [1, 4, 32, 60, 84, 4]   (Length T = 6)
      │
      ▼  [Stage 2: Token Embedding (W_E [128x16]) + Positional Encoding (W_P [12x16])]
h^(0) = W_E[ids] + W_P[0:T]        shape: [T, d_model=16]
      │
      ├──► [Stage 3: Pre-Attention LayerNorm (LN_1)]
      │    z^(1) = LN_1(h^(0))     (projects onto zero-mean sphere of exact radius √16 = 4.0)
      │
      ├──► [Stage 4: Multi-Head Causal Self-Attention (H=4 heads, d_k=4)]
      │    Q_h, K_h, V_h = split_4_heads(z^(1) @ W_{Q,K,V})
      │    A_h = softmax((Q_h @ K_h^T) / √4 + CausalMask)
      │    AttnOut = Concat(head_0, head_1, head_2, head_3) @ W_O + b_O
      │
      ▼  [First Residual Connection]
h^(1) = h^(0) + AttnOut            shape: [T, 16]
      │
      ├──► [Stage 5: Pre-FFN LayerNorm (LN_2)]
      │    z^(2) = LN_2(h^(1))     (projects onto zero-mean sphere of exact radius √16 = 4.0)
      │
      ├──► [Stage 6: Feed-Forward Key-Value Memory (d_ff=40)]
      │    m = GELU(z^(2) @ W_1 + b_1)     shape: [T, 40] (40 rule detectors)
      │    FFNOut = m @ W_2 + b_2          shape: [T, 16] (Value transitions)
      │
      ▼  [Second Residual Connection]
h^(2) = h^(1) + FFNOut             shape: [T, 16]
      │
      ▼  [Stage 7: Final LayerNorm (LN_f) & LM Head Unembedding (W_LM [16x128])]
h_final = LN_f(h^(2))
logits  = h_final[-1] @ W_LM + b_LM   shape: [vocab_size=128] (from LAST token T-1)
      │
      ▼  [Stage 8: Next-Token Prediction & Autoregressive Appending]
probs     = softmax(logits / temperature)
next_tok  = decode(probs)  ──►  "mat" (ID=96/127, P=94.39%)
updated   = ["<BOS>", "the", "cat", "sat", "on", "the"] + ["mat"]
```

---

## 3. Source of the Known Parameters (6,888 Parameters)

All parameters are constructed deterministically in [`parameter_source.py`](parameter_source.py) and exported to [`parameters.json`](parameters.json) based on four foundational Transformer interpretability papers:
- **Vaswani et al. (2017)** — *Attention Is All You Need* (Scaled Dot-Product & Multi-Head Attention)
- **Radford et al. (2019)** — *Language Models are Unsupervised Multitask Learners* (GPT-2 Pre-LN Decoder)
- **Elhage et al. (2021)** — *A Mathematical Framework for Transformer Circuits* (Attention routing heads & residual stream subspaces)
- **Geva et al. (2021)** — *Transformer Feed-Forward Layers Are Key-Value Memories* ($W_1$ as Key pattern detectors and $W_2$ as Value transition vectors)

### 3.1 The 128-Token Vocabulary Structure
| Token IDs | Count | Syntactic Category | Sample Tokens |
| :--- | ---: | :--- | :--- |
| `0 .. 3` | 4 | Control & Punctuation | `<PAD>`, `<BOS>`, `<EOS>`, `.` |
| `4 .. 11` | 8 | Determiners & Quantifiers | `the`, `a`, `an`, `this`, `that`, `every`, `some`, `each` |
| `12 .. 31` | 20 | Adjectives (5 Domains) | `small`, `quick`, `smart`, `neural`, `clinical`, `cosmic`, ... |
| `32 .. 59` | 28 | Subject Nouns (5 Domains) | `cat`, `dog`, `student`, `professor`, `model`, `transformer`, `doctor`, `telescope`, ... |
| `60 .. 83` | 24 | Verbs (5 Domains) | `sat`, `ran`, `reads`, `writes`, `predicts`, `encodes`, `treats`, `observes`, ... |
| `84 .. 95` | 12 | Prepositions | `on`, `in`, `near`, `under`, `inside`, `across`, `with`, `from`, ... |
| `96 .. 119` | 24 | Object Nouns (5 Domains) | `mat`, `garden`, `books`, `papers`, `tokens`, `vectors`, `patients`, `stars`, ... |
| `120 .. 127` | 8 | Adverbs | `quietly`, `carefully`, `accurately`, `rapidly`, `gently`, `precisely`, `clearly`, `deeply` |

### 3.2 The 16 Interpretable Embedding Dimensions (`d_model = 16`)
| Dim Range | Subspace Role | Detailed Channel Allocation |
| :--- | :--- | :--- |
| `Dims 0..6` | **Syntactic Role (Part-of-Speech)** | `0: Control`, `1: Determiner`, `2: Adjective`, `3: Subject Noun`, `4: Verb`, `5: Preposition`, `6: Object Noun / Adverb` |
| `Dims 7..11` | **Semantic Knowledge Domain** | `7: Physical/Nature`, `8: Academic/Research`, `9: AI/Computing`, `10: Medicine/Biology`, `11: Astronomy/Space` |
| `Dims 12..14` | **3-Bit Hypercube Lexical Code** | Vertices $(\pm 2.4, \pm 2.4, \pm 2.4)$ uniquely distinguish words within the same `(POS, Domain)` bucket (max pairwise cosine $= 0.8245$ across all 128 tokens) |
| `Dim 15` | **Attention-Routed Context Register** | Written dynamically by **Attention Head 1** when routing earlier prepositions (`"on"`, `"in"`) into `"the"` |

### 3.3 Complete Parameter Registry (6,888 Parameters Total)
| Parameter Group | Transformer Layer / Role | Tensor Shape | Count | Mathematical Origin |
| :--- | :--- | :--- | ---: | :--- |
| `W_E` | 1. Token Embedding Matrix | `[128, 16]` | 2,048 | Zero-centered 16D POS + Domain + 3-Bit Hypercube basis vectors |
| `W_P` | 2. Positional Encoding Matrix | `[12, 16]` | 192 | Balanced low-amplitude (`0.03`) sinusoidal position waves for `pos 0..11` |
| `ln1_gamma/beta` | 3. Pre-Attention LayerNorm | `[16], [16]` | 32 | Unit gain (`1.0`) and zero bias (`0.0`); projects onto radius-`4.0` sphere |
| `W_Q, b_Q` | 4a. Attention Query Projection | `[16, 16], [16]` | 272 | 4 Heads (`d_k=4`): Syntax (`H0`), Prep Router (`H1`), Domain (`H2`), Lexical (`H3`) |
| `W_K, b_K` | 4b. Attention Key Projection | `[16, 16], [16]` | 272 | `H1` (`col 4`) exposes high Key (`+4.0`) on Prepositions (`dim 5`) |
| `W_V, b_V` | 4c. Attention Value Projection | `[16, 16], [16]` | 272 | `H1` extracts Preposition presence (`dim 5`) and identity (`dims 7..8` for `'on'` vs `'in'`) |
| `W_O, b_O` | 4d. Attention Output Projection | `[16, 16], [16]` | 272 | Writes `H1` routed Preposition signal into `dim 15` (Context Register) and `dims 7..8` |
| `ln2_gamma/beta` | 5. Pre-FFN LayerNorm | `[16], [16]` | 32 | Projects post-attention residual stream onto the exact radius-`4.0` sphere |
| `W_1, b_1` | 6a. FFN Key Detector Matrix | `[16, 40], [40]` | 680 | 40 Spherical matched filters ($6.0 \cdot \hat{u}_j$) with cosine-cone bias ($b_1 = -21.60$) |
| `W_2, b_2` | 6b. FFN Value Transition Matrix | `[40, 16], [16]` | 656 | Writes $8.0 \cdot W_E[\text{target}_j] - h^{(1)}_j$ to steer residual stream toward next token |
| `ln_f_gamma/beta` | 7. Final LayerNorm | `[16], [16]` | 32 | Standardizes final 16D residual stream before unembedding |
| `W_LM, b_LM` | 8. Language Model Head | `[16, 128], [128]` | 2,176 | Weight-tied unit-normalized transpose $4.5 \cdot \hat{W}_E^T$ plus `<PAD>`/`<BOS>` suppression |

---

## 4. Non-Markovian Self-Attention Case Study: Three Contexts for `"the"`

In our 128-token model, the determiner `"the"` (ID `4`) appears in **three different grammatical contexts**:
1. `"<BOS> the"` $\longrightarrow$ predicts **`"cat"`** (`ID=32`, fired `N01 [Init 'the'->'cat']`, `Dim 15 = -1.78`).
2. `"<BOS> the cat sat on the"` $\longrightarrow$ **Head 1** attends `100.0%` to `"on"` (`ID=84`), writes `+15.44` into `Dim 15` and `+5.65` into `Dim 7`, fires `N28 [Routed 'on the']`, and predicts **`"mat"`** (`ID=96`, `94.39%` probability).
3. `"<BOS> a quick dog ran in the"` $\longrightarrow$ **Head 1** attends `100.0%` to `"in"` (`ID=85`), writes `+15.13` into `Dim 15` and `+5.13` into `Dim 8`, fires `N29 [Routed 'in the']`, and predicts **`"garden"`** (`ID=98`, `93.33%` probability).

Running `python3 slm_inference.py --ablation-demo` zeroes out **Attention Head 1**, proving that without Self-Attention routing the preposition into `Dim 15`, both `"on the"` and `"in the"` fail to fire `N28`/`N29` and cannot predict `"mat"` or `"garden"`.

---

## 5. How to Run

```bash
# 1. Run the full academic demonstration suite (parameter report + 8 multi-domain walkthroughs + ablation + decoding comparison)
python3 slm_inference.py

# 2. Display only the 128-token parameter source & provenance report
python3 slm_inference.py --show-parameters

# 3. Deep-inspect all internal matrices (W_E, W_P, LN, 4-Head Q/K/V, Causal Mask, 40 FFN Neurons, 128 Logits)
python3 slm_inference.py --prompt "<BOS> the cat sat on the" --inspect-step 1

# 4. Run the Causal Mechanistic Ablation experiment (disabling Attention Head 1)
python3 slm_inference.py --ablation-demo

# 5. Compare Temperature scaling (tau = 0.5, 1.0, 2.0) and Top-k filtering across the 128 tokens
python3 slm_inference.py --compare-decoding --prompt "<BOS> the cat sat on the"
```
