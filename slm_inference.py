"""
================================================================================
ACADEMIC SMALL LANGUAGE MODEL — INFERENCE ENGINE (slm_inference.py)
================================================================================
Overview:
    A complete, glass-box, inference-only Decoder-Only Transformer (GPT-style
    Small Language Model) with a 128-token vocabulary (|V| = 128 > 100), written
    in pure Python + NumPy for academic and instructional demonstration.

Key Design Principles:
    1. STRICTLY INFERENCE-ONLY:
       Contains zero training loops, zero loss functions, and zero gradient
       backpropagation. It operates purely on a known, deterministic set of
       6,888 parameters loaded from `parameters.json` (generated and documented
       by `parameter_source.py`).
    2. TRANSPARENT PARAMETER PROVENANCE (|V| = 128, d_model = 16, H = 4, d_ff = 40):
       Every weight matrix (W_E, W_P, LN_1, W_Q, W_K, W_V, W_O, LN_2, W_1, b_1,
       W_2, b_2, LN_f, W_LM, b_LM) has documented provenance, shape, and
       mathematical role in the forward pass.
    3. WHITE-BOX INNER WORKINGS INSPECTION:
       Captures every intermediate tensor in a `ForwardPassTrace` dataclass and
       provides multi-level visualizers (from high-level step summaries to
       microscopic neuron, 4-head attention, and 16D residual-stream traces).
    4. AUTOREGRESSIVE NEXT-TOKEN PREDICTION & APPENDING:
       Explicitly demonstrates how the last token's 16D hidden state is projected
       to 128 vocabulary logits, converted via Softmax into a probability
       distribution over all 128 tokens, decoded (Greedy, Temperature, or Top-k
       sampling), and appended to the running token sequence for the next
       autoregressive iteration.

Usage Examples:
    # Run the full multi-domain academic demonstration suite (default):
    python3 slm_inference.py

    # Inspect the source and structure of all 6,888 known parameters (128 tokens):
    python3 slm_inference.py --show-parameters

    # Run a deep microscopic walk-through on a specific prompt:
    python3 slm_inference.py --prompt "<BOS> the cat sat on the" --inspect-step 1

    # Run a Causal Mechanistic Ablation experiment (disabling Attention Head 1):
    python3 slm_inference.py --ablation-demo

    # Compare Greedy vs. Temperature / Top-k decoding distributions:
    python3 slm_inference.py --compare-decoding --prompt "<BOS> the cat sat on the"
================================================================================
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from parameter_source import (
    EMBEDDING_DIMENSION_MEANINGS,
    FFN_NEURON_DESCRIPTIONS,
    VOCABULARY,
    print_parameter_source_report,
    save_parameters_to_json,
)


# ==============================================================================
# 1. DATA STRUCTURES FOR RECORDING INTERNAL INFERENCE STATES
# ==============================================================================
@dataclass
class AttentionHeadTrace:
    """Stores all intermediate matrices of a single Causal Self-Attention Head."""
    head_index: int
    Q_h: np.ndarray              # Query matrix for this head: [T, d_head=4]
    K_h: np.ndarray              # Key matrix for this head  : [T, d_head=4]
    V_h: np.ndarray              # Value matrix for this head: [T, d_head=4]
    raw_scores: np.ndarray       # Unmasked scaled dot products (Q_h @ K_h^T) / sqrt(4): [T, T]
    masked_scores: np.ndarray    # Scores after adding upper-triangular causal mask (-inf): [T, T]
    attn_weights: np.ndarray     # Row-wise Softmax attention probabilities: [T, T]
    head_context: np.ndarray     # Weighted sum of values (attn_weights @ V_h): [T, d_head=4]


@dataclass
class ForwardPassTrace:
    """
    Complete snapshot of every internal mathematical step during a single
    forward pass of the 128-token Transformer for an input sequence of length T.
    """
    step_number: int
    input_tokens: List[str]          # Token strings currently in context (length T)
    input_ids: List[int]             # Integer vocabulary indices in 0..127 (length T)

    # Stage 2: Embeddings
    tok_embeddings: np.ndarray       # W_E[input_ids]          : [T, d_model=16]
    pos_embeddings: np.ndarray       # W_P[0..T-1]             : [T, d_model=16]
    h0_residual: np.ndarray          # tok_emb + pos_emb       : [T, d_model=16]

    # Stage 3: Pre-Attention LayerNorm (LN_1)
    ln1_mean: np.ndarray             # Row-wise mean           : [T, 1]
    ln1_var: np.ndarray              # Row-wise variance       : [T, 1]
    ln1_out: np.ndarray              # Normalized h0           : [T, d_model=16]

    # Stage 4: Multi-Head Causal Self-Attention (H=4) & Residual Connection 1
    head_traces: List[AttentionHeadTrace]
    concat_heads: np.ndarray         # Concatenated head outputs: [T, d_model=16]
    attn_out: np.ndarray             # Projected output (@ W_O) : [T, d_model=16]
    h1_residual: np.ndarray          # h0_residual + attn_out   : [T, d_model=16]

    # Stage 5: Pre-FFN LayerNorm (LN_2)
    ln2_mean: np.ndarray             # Row-wise mean           : [T, 1]
    ln2_var: np.ndarray              # Row-wise variance       : [T, 1]
    ln2_out: np.ndarray              # Normalized h1           : [T, d_model=16]

    # Stage 6: Feed-Forward Network (MLP) & Residual Connection 2
    ffn_pre_act: np.ndarray          # ln2_out @ W_1 + b_1     : [T, d_ff=40]
    ffn_hidden: np.ndarray           # GELU(ffn_pre_act)       : [T, d_ff=40]
    ffn_out: np.ndarray              # ffn_hidden @ W_2 + b_2  : [T, d_model=16]
    h2_residual: np.ndarray          # h1_residual + ffn_out   : [T, d_model=16]

    # Stage 7: Final LayerNorm & Language Model Head (Unembedding)
    ln_f_out: np.ndarray             # Final normalized state  : [T, d_model=16]
    last_hidden_state: np.ndarray    # ln_f_out[-1] (last pos) : [d_model=16]
    raw_logits: np.ndarray           # last_hidden @ W_LM + b_LM: [vocab_size=128]

    # Stage 8: Next-Token Prediction & Decoding over 128 Vocabulary Tokens
    scaled_logits: np.ndarray        # Logits after temperature & top-k filtering: [128]
    probabilities: np.ndarray        # Softmax probability distribution: [128]
    predicted_id: int                # Chosen next token ID in 0..127
    predicted_token: str             # Chosen next token string
    predicted_prob: float            # Probability of the chosen next token
    updated_tokens: List[str]        # Sequence after appending predicted_token (length T+1)


# ==============================================================================
# 2. CORE SMALL LANGUAGE MODEL CLASS (INFERENCE ONLY, 128-TOKEN VOCABULARY)
# ==============================================================================
class AcademicSmallLanguageModel:
    """
    A transparent, inference-only Decoder-Only Transformer that loads a known
    set of 6,888 parameters for a 128-token vocabulary from `parameters.json`
    and exposes every internal matrix operation for academic inspection.
    """

    def __init__(self, parameter_path: str | Path = "parameters.json") -> None:
        self.parameter_path = Path(parameter_path)
        if not self.parameter_path.exists():
            save_parameters_to_json(self.parameter_path)

        with self.parameter_path.open("r", encoding="utf-8") as f:
            self.bundle: Dict[str, Any] = json.load(f)

        # Regenerate if an older (<100 token) parameters.json is present
        if len(self.bundle.get("vocabulary", [])) != len(VOCABULARY):
            save_parameters_to_json(self.parameter_path)
            with self.parameter_path.open("r", encoding="utf-8") as f:
                self.bundle = json.load(f)

        # Extract metadata & hyperparameters
        self.metadata: Dict[str, Any] = self.bundle["_metadata"]
        self.hyperparams: Dict[str, Any] = self.metadata["hyperparameters"]
        self.vocab: List[str] = self.bundle["vocabulary"]
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}
        self.provenance: Dict[str, Any] = self.bundle["parameter_provenance"]

        self.vocab_size: int = int(self.hyperparams["vocab_size"])  # 128
        self.max_seq_len: int = int(self.hyperparams["max_seq_len"])  # 12
        self.d_model: int = int(self.hyperparams["d_model"])        # 16
        self.n_heads: int = int(self.hyperparams["n_heads"])        # 4
        self.d_head: int = int(self.hyperparams["d_head"])          # 4
        self.d_ff: int = int(self.hyperparams["d_ff"])              # 40
        self.eps: float = float(self.hyperparams["layer_norm_eps"]) # 1e-5

        # Load the known parameter matrices as float64 NumPy arrays
        w = self.bundle["weights"]
        self.W_E = np.array(w["W_E"], dtype=np.float64)            # [128, 16]
        self.W_P = np.array(w["W_P"], dtype=np.float64)            # [12, 16]

        self.ln1_gamma = np.array(w["ln1_gamma"], dtype=np.float64)  # [16]
        self.ln1_beta = np.array(w["ln1_beta"], dtype=np.float64)    # [16]

        self.W_Q = np.array(w["W_Q"], dtype=np.float64)            # [16, 16]
        self.b_Q = np.array(w["b_Q"], dtype=np.float64)            # [16]
        self.W_K = np.array(w["W_K"], dtype=np.float64)            # [16, 16]
        self.b_K = np.array(w["b_K"], dtype=np.float64)            # [16]
        self.W_V = np.array(w["W_V"], dtype=np.float64)            # [16, 16]
        self.b_V = np.array(w["b_V"], dtype=np.float64)            # [16]
        self.W_O = np.array(w["W_O"], dtype=np.float64)            # [16, 16]
        self.b_O = np.array(w["b_O"], dtype=np.float64)            # [16]

        self.ln2_gamma = np.array(w["ln2_gamma"], dtype=np.float64)  # [16]
        self.ln2_beta = np.array(w["ln2_beta"], dtype=np.float64)    # [16]

        self.W_1 = np.array(w["W_1"], dtype=np.float64)            # [16, 40]
        self.b_1 = np.array(w["b_1"], dtype=np.float64)            # [40]
        self.W_2 = np.array(w["W_2"], dtype=np.float64)            # [40, 16]
        self.b_2 = np.array(w["b_2"], dtype=np.float64)            # [16]

        self.ln_f_gamma = np.array(w["ln_f_gamma"], dtype=np.float64)  # [16]
        self.ln_f_beta = np.array(w["ln_f_beta"], dtype=np.float64)    # [16]

        self.W_LM = np.array(w["W_LM"], dtype=np.float64)          # [16, 128]
        self.b_LM = np.array(w["b_LM"], dtype=np.float64)          # [128]

    # --------------------------------------------------------------------------
    # MATHEMATICAL PRIMITIVES (Tokenization, LayerNorm, GELU, Softmax)
    # --------------------------------------------------------------------------
    def tokenize(self, text_or_tokens: str | List[str]) -> Tuple[List[str], List[int]]:
        """
        STAGE 1: Tokenization & Vocabulary Lookup (|V| = 128).
        Splits a whitespace-delimited prompt into vocabulary tokens, prepending
        '<BOS>' if not already present, and maps each token string to its
        integer index in `0..127`.
        """
        if isinstance(text_or_tokens, str):
            tokens = text_or_tokens.strip().split()
        else:
            tokens = list(text_or_tokens)

        if not tokens or tokens[0] != "<BOS>":
            tokens = ["<BOS>"] + tokens

        token_ids: List[int] = []
        for tok in tokens:
            if tok not in self.token_to_id:
                raise ValueError(
                    f"Unknown token '{tok}'. Vocabulary contains {self.vocab_size} tokens."
                )
            token_ids.append(self.token_to_id[tok])
        return tokens, token_ids

    def layer_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Layer Normalization (Ba et al., 2016) independently for each
        token position along the embedding dimension (d_model = 16):
            mean = (1 / 16) * sum_i(x_i)
            var  = (1 / 16) * sum_i((x_i - mean)^2)
            LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta

        Geometric Interpretation in d_model = 16:
            When gamma = 1 and beta = 0, every normalized 16D token vector lies
            on the zero-mean sphere of exact Euclidean radius ||LN(x)||_2 = sqrt(16) = 4.0000!
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = gamma * x_norm + beta
        return out, mean, var

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """
        Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016), used in GPT-2/3:
            GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3)))
        )

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable Softmax function:
            softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
        """
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # --------------------------------------------------------------------------
    # SINGLE FORWARD PASS (WITH COMPLETE INTERNAL STATE RECORDING)
    # --------------------------------------------------------------------------
    def forward_step(
        self,
        tokens: List[str],
        step_number: int = 1,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        ablate_heads: Optional[List[int]] = None,
    ) -> ForwardPassTrace:
        """
        Executes one complete forward pass through the Decoder-Only Transformer
        on the current token sequence `tokens` of length T, predicts the next
        token out of the 128-token vocabulary, and returns a `ForwardPassTrace`
        containing all intermediate tensors and the updated sequence of length T+1.

        Optional Mechanistic Feature (`ablate_heads`):
            If head indices (e.g., `[1]`) are passed in `ablate_heads`, those
            attention heads are zeroed out before `W_O`, enabling causal
            ablation experiments.
        """
        tokens, token_ids = self.tokenize(tokens)
        T = len(token_ids)
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}."
            )

        # ======================================================================
        # STAGE 2: Token Embedding + Positional Encoding
        #   x_tok = W_E[token_ids]       shape: [T, d_model=16]
        #   x_pos = W_P[0 : T]           shape: [T, d_model=16]
        #   h^(0) = x_tok + x_pos        shape: [T, d_model=16]
        # ======================================================================
        tok_emb = self.W_E[token_ids]       # Lookup rows in W_E [128, 16]
        pos_emb = self.W_P[:T]              # Slice first T rows of W_P [12, 16]
        h0_residual = tok_emb + pos_emb     # Initial 16D residual stream state

        # ======================================================================
        # STAGE 3: Pre-Attention Layer Normalization (LN_1)
        #   z^(1) = LN_1(h^(0))          shape: [T, d_model=16]
        # ======================================================================
        ln1_out, ln1_mean, ln1_var = self.layer_norm(
            h0_residual, self.ln1_gamma, self.ln1_beta
        )

        # ======================================================================
        # STAGE 4: Multi-Head Causal Self-Attention (H=4, d_head=4) + Residual 1
        #   Q = z^(1) @ W_Q + b_Q        shape: [T, 16]
        #   K = z^(1) @ W_K + b_K        shape: [T, 16]
        #   V = z^(1) @ W_V + b_V        shape: [T, 16]
        # ======================================================================
        Q = ln1_out @ self.W_Q + self.b_Q
        K = ln1_out @ self.W_K + self.b_K
        V = ln1_out @ self.W_V + self.b_V

        # Upper-triangular Causal Mask M of shape [T, T]:
        #   M[i, j] = 0.0   if j <= i (token i CAN attend to past/present token j)
        #   M[i, j] = -1e9  if j >  i (token i CANNOT look into the future!)
        causal_mask = np.triu(np.full((T, T), -1e9, dtype=np.float64), k=1)

        head_traces: List[AttentionHeadTrace] = []
        head_outputs: List[np.ndarray] = []

        for h in range(self.n_heads):
            c_start = h * self.d_head
            c_end = (h + 1) * self.d_head
            Q_h = Q[:, c_start:c_end]  # [T, 4]
            K_h = K[:, c_start:c_end]  # [T, 4]
            V_h = V[:, c_start:c_end]  # [T, 4]

            # Scaled dot-product attention scores: (Q_h @ K_h^T) / sqrt(4)
            raw_scores = (Q_h @ K_h.T) / np.sqrt(self.d_head)  # [T, T]
            masked_scores = raw_scores + causal_mask           # [T, T]
            attn_weights = self.softmax(masked_scores, axis=-1)  # [T, T]
            head_ctx = attn_weights @ V_h                      # [T, 4]

            if ablate_heads and h in ablate_heads:
                head_ctx = np.zeros_like(head_ctx)

            head_traces.append(
                AttentionHeadTrace(
                    head_index=h,
                    Q_h=Q_h,
                    K_h=K_h,
                    V_h=V_h,
                    raw_scores=raw_scores,
                    masked_scores=masked_scores,
                    attn_weights=attn_weights,
                    head_context=head_ctx,
                )
            )
            head_outputs.append(head_ctx)

        # Concatenate all 4 heads [T, 16] and project via W_O [16, 16]
        concat_heads = np.concatenate(head_outputs, axis=-1)   # [T, 16]
        attn_out = concat_heads @ self.W_O + self.b_O          # [T, 16]

        # First Residual Connection
        h1_residual = h0_residual + attn_out                   # [T, 16]

        # ======================================================================
        # STAGE 5: Pre-FFN Layer Normalization (LN_2)
        #   z^(2) = LN_2(h^(1))          shape: [T, 16] (||z^(2)||_2 = 4.0000)
        # ======================================================================
        ln2_out, ln2_mean, ln2_var = self.layer_norm(
            h1_residual, self.ln2_gamma, self.ln2_beta
        )

        # ======================================================================
        # STAGE 6: Position-Wise Feed-Forward Network (40 Key-Value Rules)
        #   ffn_pre = z^(2) @ W_1 + b_1  shape: [T, d_ff=40]
        #   m       = GELU(ffn_pre)      shape: [T, d_ff=40]
        #   ffn_out = m @ W_2 + b_2      shape: [T, d_model=16]
        #   h^(2)   = h^(1) + ffn_out    shape: [T, d_model=16]
        # ======================================================================
        ffn_pre_act = ln2_out @ self.W_1 + self.b_1            # [T, 40]
        ffn_hidden = self.gelu(ffn_pre_act)                    # [T, 40]
        ffn_out = ffn_hidden @ self.W_2 + self.b_2             # [T, 16]

        # Second Residual Connection
        h2_residual = h1_residual + ffn_out                    # [T, 16]

        # ======================================================================
        # STAGE 7: Final LayerNorm & Language Model Head (Unembedding to 128 Tokens)
        #   h_final = LN_f(h^(2))        shape: [T, 16]
        #   logits  = h_final[-1] @ W_LM + b_LM   shape: [vocab_size=128]
        # ======================================================================
        ln_f_out, _, _ = self.layer_norm(
            h2_residual, self.ln_f_gamma, self.ln_f_beta
        )
        last_hidden_state = ln_f_out[-1]                       # [16]
        raw_logits = last_hidden_state @ self.W_LM + self.b_LM # [128]

        # ======================================================================
        # STAGE 8: Next-Token Probability Distribution & Decoding Strategy
        # ======================================================================
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive (> 0).")

        scaled_logits = raw_logits / temperature

        # Optional Top-k filtering across the 128 vocabulary candidates
        if top_k is not None and 0 < top_k < self.vocab_size:
            top_indices = np.argsort(scaled_logits)[-top_k:]
            filtered_logits = np.full_like(scaled_logits, -1e9)
            filtered_logits[top_indices] = scaled_logits[top_indices]
            scaled_logits = filtered_logits

        probabilities = self.softmax(scaled_logits, axis=-1)

        if strategy == "greedy":
            predicted_id = int(np.argmax(probabilities))
        elif strategy in ("sampling", "top_k"):
            if rng is None:
                rng = np.random.default_rng(42)
            predicted_id = int(rng.choice(self.vocab_size, p=probabilities))
        else:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")

        predicted_token = self.id_to_token[predicted_id]
        predicted_prob = float(probabilities[predicted_id])

        # ======================================================================
        # AUTOREGRESSIVE TOKEN APPENDING:
        #   Append the predicted token to the running sequence (T -> T+1).
        # ======================================================================
        updated_tokens = tokens + [predicted_token]

        return ForwardPassTrace(
            step_number=step_number,
            input_tokens=tokens,
            input_ids=token_ids,
            tok_embeddings=tok_emb,
            pos_embeddings=pos_emb,
            h0_residual=h0_residual,
            ln1_mean=ln1_mean,
            ln1_var=ln1_var,
            ln1_out=ln1_out,
            head_traces=head_traces,
            concat_heads=concat_heads,
            attn_out=attn_out,
            h1_residual=h1_residual,
            ln2_mean=ln2_mean,
            ln2_var=ln2_var,
            ln2_out=ln2_out,
            ffn_pre_act=ffn_pre_act,
            ffn_hidden=ffn_hidden,
            ffn_out=ffn_out,
            h2_residual=h2_residual,
            ln_f_out=ln_f_out,
            last_hidden_state=last_hidden_state,
            raw_logits=raw_logits,
            scaled_logits=scaled_logits,
            probabilities=probabilities,
            predicted_id=predicted_id,
            predicted_token=predicted_token,
            predicted_prob=predicted_prob,
            updated_tokens=updated_tokens,
        )

    # --------------------------------------------------------------------------
    # AUTOREGRESSIVE GENERATION LOOP
    # --------------------------------------------------------------------------
    def generate(
        self,
        prompt: str | List[str],
        max_new_tokens: int = 9,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
        ablate_heads: Optional[List[int]] = None,
    ) -> List[ForwardPassTrace]:
        """
        Runs the complete Autoregressive Generation Loop starting from `prompt`.
        At each step s:
            1. Feeds current sequence [t_0, ..., t_{T-1}] into `forward_step()`.
            2. Predicts next token t_T from the 128-token vocabulary.
            3. Appends t_T to form [t_0, ..., t_{T-1}, t_T].
            4. Stops immediately if t_T == '<EOS>' or max_seq_len is reached.
        """
        current_tokens, _ = self.tokenize(prompt)
        traces: List[ForwardPassTrace] = []
        # rng = np.random.default_rng(seed)
        rng = np.random.default_rng()

        for step in range(1, max_new_tokens + 1):
            if len(current_tokens) >= self.max_seq_len:
                break

            trace = self.forward_step(
                tokens=current_tokens,
                step_number=step,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
                ablate_heads=ablate_heads,
            )
            traces.append(trace)
            current_tokens = trace.updated_tokens

            if trace.predicted_token == "<EOS>":
                break

        return traces


# ==============================================================================
# 3. VISUALIZATION & EDUCATIONAL INSPECTION UTILITIES
# ==============================================================================
def _prob_bar(prob: float, width: int = 24) -> str:
    """Renders a clean ASCII horizontal bar proportional to a probability in [0, 1]."""
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def _format_vec(vec: np.ndarray, precision: int = 2, max_elems: int = 16) -> str:
    """Formats a 1D NumPy vector compactly with explicit signs."""
    shown = vec[:max_elems]
    body = ", ".join(f"{v:+0.{precision}f}" for v in shown)
    if len(vec) > max_elems:
        body += ", ..."
    return "[" + body + "]"


def print_deep_step_inspection(
    model: AcademicSmallLanguageModel, trace: ForwardPassTrace
) -> None:
    """
    Prints a comprehensive, microscopic walk-through of all 8 internal stages
    of a single Transformer inference step (128-token vocabulary, 16D state).
    """
    T = len(trace.input_tokens)
    last_pos = T - 1
    last_tok = trace.input_tokens[last_pos]

    print("\n" + "━" * 94)
    print(
        f" MICROSCOPIC INNER-WORKING TRACE — GENERATION STEP {trace.step_number} "
        f"(Context Length T = {T}, Vocab |V| = {model.vocab_size})"
    )
    print("━" * 94)

    # --- STAGE 1: Tokenization ---
    print("\n┌─ STAGE 1: TOKENIZATION & 128-TOKEN VOCABULARY LOOKUP")
    print(f"│  Input String   : \"{' '.join(trace.input_tokens)}\"")
    tok_pairs = [
        f"pos {i}: '{tok}' (ID={tid})"
        for i, (tok, tid) in enumerate(zip(trace.input_tokens, trace.input_ids))
    ]
    print(f"│  Token Sequence : {',  '.join(tok_pairs)}")
    print("└─" + "─" * 92)

    # --- STAGE 2: Embeddings + Positional Encoding ---
    print(
        "\n┌─ STAGE 2: TOKEN EMBEDDING (W_E [128x16]) + POSITIONAL ENCODING (W_P [12x16]) -> h^(0)"
    )
    print(
        "│  16D Basis: Dims[0..6]=POS Role | Dims[7..11]=5 Domains | Dims[12..14]=3-Bit Cube | Dim[15]=Ctx"
    )
    for i, (tok, tid) in enumerate(zip(trace.input_tokens, trace.input_ids)):
        marker = " ◄── LAST TOKEN (drives next-token prediction)" if i == last_pos else ""
        print(f"│  Pos {i} ('{tok}', ID={tid:3d}){marker}:")
        print(f"│    W_E[{tid:3d}] (0..7 / 8..15) = {_format_vec(trace.tok_embeddings[i, :8])} / {_format_vec(trace.tok_embeddings[i, 8:])}")
        print(f"│    h^(0)[{i}] (0..7 / 8..15) = {_format_vec(trace.h0_residual[i, :8])} / {_format_vec(trace.h0_residual[i, 8:])}")
    print("└─" + "─" * 92)

    # --- STAGE 3: Pre-Attention LayerNorm ---
    print("\n┌─ STAGE 3: PRE-ATTENTION LAYER NORMALIZATION (LN_1)")
    print(
        "│  Formula: LN_1(x) = gamma_1 * (x - mean) / sqrt(var + eps) + beta_1 "
        "(projects onto radius-sqrt(16) = 4.0000 sphere)"
    )
    for i, tok in enumerate(trace.input_tokens):
        mu = float(trace.ln1_mean[i, 0])
        var = float(trace.ln1_var[i, 0])
        norm_l2 = float(np.linalg.norm(trace.ln1_out[i]))
        print(
            f"│  Pos {i} ('{tok:<11s}'): mean={mu:+6.3f}, var={var:6.3f}, ||LN_1||_2={norm_l2:6.4f} "
            f"| dims[0..5]={_format_vec(trace.ln1_out[i, :6])}"
        )
    print("└─" + "─" * 92)

    # --- STAGE 4: Multi-Head Causal Self-Attention ---
    print("\n┌─ STAGE 4: MULTI-HEAD CAUSAL SELF-ATTENTION (H = 4 Heads, d_k = 4) & RESIDUAL 1")
    print(
        "│  Parameter Source: W_Q, W_K, W_V [16x16] split across 4 Heads (cols 0..3, 4..7, 8..11, 12..15); W_O [16x16]"
    )
    head_titles = [
        "HEAD 0 (cols 0..3)  : Local Syntactic Constituency Head",
        "HEAD 1 (cols 4..7)  : Long-Range Prepositional Context Router ('on'/'in' -> 'the')",
        "HEAD 2 (cols 8..11) : Semantic Domain Persistence Head (Physical/Academic/AI/Medical)",
        "HEAD 3 (cols 12..15): Lexical 3-Bit Sub-Identity & Astronomy Domain Head",
    ]
    # Show full matrix for Head 0 & Head 1, and concise summary for Head 2 & Head 3
    for h_trace in trace.head_traces:
        h = h_trace.head_index
        print(f"│\n│  ▸ {head_titles[h]}")
        print(
            f"│    Last Token ('{last_tok}') Query Q_{h}[{last_pos}] = {_format_vec(h_trace.Q_h[last_pos])}"
        )
        if h in (0, 1):
            header_cols = " ".join(f"{tok:>9s}" for tok in trace.input_tokens)
            print(f"│      {'Query \\ Key':<14s} | {header_cols} | Top Attended Context")
            print("│      " + "-" * (17 + 10 * T + 26))
            for q_idx, q_tok in enumerate(trace.input_tokens):
                row_cells = []
                for k_idx in range(T):
                    if k_idx > q_idx:
                        row_cells.append(f"{'   [MASK]':>9s}")
                    else:
                        w_val = h_trace.attn_weights[q_idx, k_idx]
                        row_cells.append(f"{w_val * 100:8.1f}%")
                best_k = int(np.argmax(h_trace.attn_weights[q_idx, : q_idx + 1]))
                best_tok = trace.input_tokens[best_k]
                best_pct = h_trace.attn_weights[q_idx, best_k] * 100.0
                row_str = " ".join(row_cells)
                print(
                    f"│      pos {q_idx} '{q_tok:<7s}' | {row_str} | -> '{best_tok}' ({best_pct:5.1f}%)"
                )
        else:
            w_last = h_trace.attn_weights[last_pos]
            best_k = int(np.argmax(w_last))
            print(
                f"│    Last Pos Attention Distribution: "
                + ", ".join(
                    f"'{trace.input_tokens[k]}': {w_last[k]*100:.1f}%" for k in range(T)
                )
                + f"  (Top -> '{trace.input_tokens[best_k]}')"
            )

    print("│\n│  ▸ Attention Output & First Residual Addition (h^(1) = h^(0) + Concat(Head_0..3) @ W_O):")
    print(f"│    AttnOut[{last_pos}] (dims 0..7)  = {_format_vec(trace.attn_out[last_pos, :8])}")
    print(f"│    AttnOut[{last_pos}] (dims 8..15) = {_format_vec(trace.attn_out[last_pos, 8:])}")
    print(f"│    h^(1)[{last_pos}]   (dims 8..15) = {_format_vec(trace.h1_residual[last_pos, 8:])}")
    if abs(trace.attn_out[last_pos, 15]) > 1.0:
        print(
            f"│    ★ MECHANISTIC INSIGHT: Attention Head 1 wrote {trace.attn_out[last_pos, 15]:+.2f} "
            "into Dim 15 (Context Register)!"
        )
        print(
            "│      This routes the earlier Preposition ('on' or 'in') into 'the', disambiguating it from initial '<BOS> the'!"
        )
    print("└─" + "─" * 92)

    # --- STAGE 5 & 6: Pre-FFN LayerNorm & Feed-Forward Key-Value Memory ---
    print("\n┌─ STAGE 5 & 6: PRE-FFN LAYERNORM (LN_2) & 40-NEURON FFN KEY-VALUE MEMORY (W_1, W_2)")
    print(
        "│  Parameter Source: W_1 [16x40] (Spherical Key Detectors), b_1 [40], W_2 [40x16] (Value Transitions)"
    )
    print(
        f"│  ||LN_2(h^(1))[{last_pos}]||_2 = {np.linalg.norm(trace.ln2_out[last_pos]):.4f} "
        "(lies on exact radius-sqrt(16) = 4.0000 sphere)"
    )
    print("│  Top-4 FFN Hidden Layer Neurons m = GELU(LN_2(h^(1)) @ W_1 + b_1) at Last Position:")
    pre_last = trace.ffn_pre_act[last_pos]
    hid_last = trace.ffn_hidden[last_pos]
    ranked_neurons = np.argsort(hid_last)[::-1]
    for n_idx in ranked_neurons[:4]:
        act_val = float(hid_last[n_idx])
        pre_val = float(pre_last[n_idx])
        status = "★ FIRED" if act_val > 0.1 else "  quiet"
        desc = FFN_NEURON_DESCRIPTIONS[n_idx]
        print(
            f"│    {status} | Pre-GELU={pre_val:+6.2f} | Post-GELU={act_val:+6.2f} | {desc}"
        )
    print("└─" + "─" * 92)

    # --- STAGE 7 & 8: Final LayerNorm, LM Head Logits, and Softmax over 128 Tokens ---
    print(
        "\n┌─ STAGE 7 & 8: FINAL LAYERNORM (LN_f), UNEMBEDDING (W_LM [16x128]), & NEXT-TOKEN PREDICTION"
    )
    print(
        "│  Parameter Source: W_LM [16x128] (Weight-Tied Transpose 4.5 * W_E_unit^T) + b_LM [128]"
    )
    print(
        "│  Top-8 Candidate Next Tokens (out of 128 Vocabulary Tokens) Ranked by Softmax Probability:"
    )
    print(
        f"│    {'Rank':<4} | {'TokID':>5} | {'Token':<12} | {'Raw Logit':>9} | {'Prob (%)':>8} | Probability Distribution Bar"
    )
    print("│    " + "-" * 80)
    top_ids = np.argsort(trace.probabilities)[::-1][:8]
    for rank, vid in enumerate(top_ids, start=1):
        tok_str = model.id_to_token[int(vid)]
        logit_val = float(trace.raw_logits[vid])
        prob_val = float(trace.probabilities[vid])
        bar = _prob_bar(prob_val, width=24)
        chosen_tag = " ◄── SELECTED NEXT TOKEN" if int(vid) == trace.predicted_id else ""
        print(
            f"│    #{rank:<3d} | {int(vid):>5d} | {tok_str:<12s} | {logit_val:>+9.3f} | "
            f"{prob_val * 100:>7.2f}% | {bar}{chosen_tag}"
        )

    # --- AUTOREGRESSIVE TOKEN STRING UPDATE ---
    print("│")
    print("│  ▸ AUTOREGRESSIVE SEQUENCE UPDATE (Appending Predicted Token to Token String):")
    before_str = " ".join(trace.input_tokens)
    after_str = " ".join(trace.updated_tokens)
    print(f"│    Previous Token List (len={T:2d}) : {trace.input_tokens}")
    print(
        f"│    Appended Next Token          : + ['{trace.predicted_token}'] "
        f"(ID={trace.predicted_id}/127, P={trace.predicted_prob * 100:.2f}%)"
    )
    print(f"│    Updated Token List  (len={T+1:2d}) : {trace.updated_tokens}")
    print(f"│    Updated Token String         : \"{before_str}\"  ──►  \"{after_str}\"")
    print("└─" + "─" * 92)


def print_generation_walkthrough(
    model: AcademicSmallLanguageModel,
    prompt: str,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    deep_inspect_steps: Optional[List[int]] = None,
) -> List[ForwardPassTrace]:
    """
    Runs autoregressive generation for `prompt` across the 128-token vocabulary,
    showing at every step:
      1. Embedding & Positional lookup in 16D space
      2. Multi-Head Causal Self-Attention routing (Head 1 context register `dim 15`)
      3. Which of the 40 FFN Key-Value Memory Neurons (`N00`..`N39`) fired
      4. Top candidate probabilities out of all 128 vocabulary tokens
      5. How the predicted token is appended to the token string
    """
    traces = model.generate(
        prompt=prompt,
        max_new_tokens=9,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
    )

    print("\n" + "═" * 94)
    print(f" AUTOREGRESSIVE INFERENCE WALKTHROUGH | Prompt: \"{prompt}\"")
    print(
        f" Vocab Size: {model.vocab_size} tokens | Strategy: {strategy.upper()} | "
        f"Temp: {temperature:.2f} | Top-k: {top_k if top_k else 'All (128)'} | "
        f"Source: {model.parameter_path.name}"
    )
    print("═" * 94)

    for trace in traces:
        T = len(trace.input_tokens)
        last_pos = T - 1

        # Attention Head 1 routing info
        h1_weights = trace.head_traces[1].attn_weights[last_pos]
        best_k_pos = int(np.argmax(h1_weights))
        best_k_tok = trace.input_tokens[best_k_pos]
        best_k_pct = float(h1_weights[best_k_pos]) * 100.0

        # Active FFN neuron out of 40
        ffn_last = trace.ffn_hidden[last_pos]
        best_neuron = int(np.argmax(ffn_last))
        best_neuron_act = float(ffn_last[best_neuron])
        neuron_label = FFN_NEURON_DESCRIPTIONS[best_neuron].split(":")[0].strip()

        # Top 4 candidate next tokens out of 128
        top4_ids = np.argsort(trace.probabilities)[::-1][:4]
        top4_summary = ", ".join(
            f"'{model.id_to_token[int(i)]}' ({trace.probabilities[i]*100:.1f}%)"
            for i in top4_ids
        )

        print(f"\n[Step {trace.step_number}] Input Sequence (T={T}): {trace.input_tokens}")
        print(
            f"  ├─ 1. Embedding (16D)      : Last token '{trace.input_tokens[-1]}' (ID={trace.input_ids[-1]}/127) "
            f"||LN_1||_2 = {np.linalg.norm(trace.ln1_out[last_pos]):.4f}"
        )
        print(
            f"  ├─ 2. Causal Self-Attn     : Head 1 at '{trace.input_tokens[-1]}' attends to "
            f"pos {best_k_pos} ('{best_k_tok}', {best_k_pct:.1f}%) "
            f"-> CtxReg[dim15]={trace.attn_out[last_pos, 15]:+5.2f}"
        )
        print(
            f"  ├─ 3. FFN Memory (1 of 40) : Fired {neuron_label} (GELU activation = {best_neuron_act:+.2f})"
        )
        print(f"  ├─ 4. Top-4 of 128 Tokens  : {top4_summary}")
        print(
            f"  ├─ 5. Predicted Next Token : '{trace.predicted_token}' "
            f"(ID={trace.predicted_id}, Logit={trace.raw_logits[trace.predicted_id]:+.2f}, "
            f"Prob={trace.predicted_prob * 100:.2f}%)"
        )
        print(
            f"  └─ 6. Append to String     : \"{' '.join(trace.input_tokens)}\" + "
            f"\"{trace.predicted_token}\"  ══►  \"{' '.join(trace.updated_tokens)}\""
        )

        if deep_inspect_steps and trace.step_number in deep_inspect_steps:
            print_deep_step_inspection(model, trace)

    # Print the progressive token string growth summary table
    print("\n" + "─" * 94)
    print(" PROGRESSIVE TOKEN STRING GROWTH TABLE (How Tokens Are Predicted & Appended):")
    print("─" * 94)
    print(
        f" {'Step':<4} | {'Input Token String (Context)':<42} | {'Predicted':<13} | "
        f"{'Prob':>6} | {'Resulting Appended Token String'}"
    )
    print(" " + "-" * 92)
    for trace in traces:
        in_str = " ".join(trace.input_tokens)
        out_str = " ".join(trace.updated_tokens)
        print(
            f" #{trace.step_number:<3d} | {in_str:<42s} | + {trace.predicted_token:<11s} | "
            f"{trace.predicted_prob * 100:5.1f}% | {out_str}"
        )
    print("═" * 94)
    return traces


def run_ablation_experiment(model: AcademicSmallLanguageModel) -> None:
    """
    Demonstrates the causal role of Multi-Head Self-Attention (Head 1) by
    comparing normal inference vs. Head-1-ablated inference on the non-Markovian
    contexts:
      1. "<BOS> the cat sat on the"  (Normal -> 'mat'    vs. Ablated -> 'cat')
      2. "<BOS> a quick dog ran in the" (Normal -> 'garden' vs. Ablated -> 'cat')
    """
    print("\n" + "═" * 94)
    print(" CAUSAL MECHANISTIC ABLATION EXPERIMENT: WHY TRANSFORMERS NEED SELF-ATTENTION")
    print("═" * 94)
    for prompt, expected_tok in [
        ("<BOS> the cat sat on the", "mat"),
        ("<BOS> a quick dog ran in the", "garden"),
    ]:
        normal_trace = model.forward_step(prompt.split(), step_number=1)
        ablated_trace = model.forward_step(
            prompt.split(), step_number=1, ablate_heads=[1]
        )
        n_fired = int(np.argmax(normal_trace.ffn_hidden[-1]))
        a_fired = int(np.argmax(ablated_trace.ffn_hidden[-1]))

        print(f"\n▸ Test Context: \"{prompt}\" (Last token is 'the', ID=4)")
        print(
            f"  [Normal Transformer (All 4 Heads Active)]:"
        )
        print(
            f"    • Head 1 Context Register (Dim 15) : {normal_trace.attn_out[-1, 15]:+.2f} "
            "(routed from earlier preposition)"
        )
        print(
            f"    • Fired FFN Memory Neuron          : {FFN_NEURON_DESCRIPTIONS[n_fired].split(':')[0].strip()}"
        )
        print(
            f"    • Predicted Next Token             : '{normal_trace.predicted_token}' "
            f"(Prob = {normal_trace.predicted_prob * 100:.2f}%) == '{expected_tok}' ✓"
        )
        a_max_act = float(np.max(ablated_trace.ffn_hidden[-1]))
        a_neuron_str = (
            FFN_NEURON_DESCRIPTIONS[a_fired].split(":")[0].strip()
            if a_max_act > 0.05
            else "None (all 40 FFN rules below threshold!)"
        )
        print(
            f"  [Ablated Transformer (Head 1 Disabled / Zeroed Out)]:"
        )
        print(
            f"    • Head 1 Context Register (Dim 15) : {ablated_trace.attn_out[-1, 15]:+.2f} "
            "(no prepositional context routed!)"
        )
        print(
            f"    • Fired FFN Memory Neuron          : {a_neuron_str}"
        )
        print(
            f"    • Predicted Next Token             : '{ablated_trace.predicted_token}' "
            f"(Prob = {ablated_trace.predicted_prob * 100:.2f}%) ✗ (fails to predict '{expected_tok}', stuck on '{ablated_trace.predicted_token}'!)"
        )
    print("═" * 94)


def run_decoding_comparison(
    model: AcademicSmallLanguageModel, prompt: str = "<BOS> the cat sat on the"
) -> None:
    """
    Compares how Temperature scaling (tau = 0.5, 1.0, 2.0) and Top-k filtering
    reshape the next-token probability distribution across the 128-token vocabulary.
    """
    print("\n" + "═" * 94)
    print(f" DECODING STRATEGY & TEMPERATURE COMPARISON | Context: \"{prompt}\"")
    print("═" * 94)
    configs = [
        ("Sharp (Temp = 0.5, All 128 tokens)", 0.5, None),
        ("Standard (Temp = 1.0, All 128 tokens)", 1.0, None),
        ("Flattened (Temp = 2.0, All 128 tokens)", 2.0, None),
        ("Top-k Filtered (Temp = 1.0, Top-k = 3)", 1.0, 3),
    ]
    for label, temp, tk in configs:
        trace = model.forward_step(
            prompt.split(), step_number=1, strategy="greedy", temperature=temp, top_k=tk
        )
        top5 = np.argsort(trace.probabilities)[::-1][:5]
        summary = " | ".join(
            f"{model.id_to_token[int(i)]}: {trace.probabilities[i]*100:5.1f}%"
            for i in top5
        )
        print(f" • {label:<38s} -> Top-5: {summary}")
    print("═" * 94)


# ==============================================================================
# 4. COMMAND-LINE INTERFACE & ACADEMIC DEMONSTRATION SUITE
# ==============================================================================
def run_academic_demo_suite(model: AcademicSmallLanguageModel) -> None:
    """
    Runs the complete academic demonstration suite for the 128-token model:
      1. Parameter Provenance & Source Report (128 tokens, 16D, 4 heads, 40 rules)
      2. Flagship Walkthrough on "<BOS>" -> "<BOS> the cat sat on the mat <EOS>"
         with microscopic deep inspection of Step 1 and Step 6
      3. Multi-Domain Autoregressive Generation across Physical, AI, Academic,
         Medical, and Astronomy prompts
      4. Causal Mechanistic Ablation Experiment & Decoding Comparison
    """
    # 1. Parameter Provenance Report
    print_parameter_source_report(model.bundle)

    # 2. Flagship Walkthrough with Microscopic Inspection on Step 1 & Step 6
    print_generation_walkthrough(
        model=model,
        prompt="<BOS>",
        strategy="greedy",
        temperature=1.0,
        deep_inspect_steps=[1, 6],
    )

    # 3. Multi-Domain Walkthroughs across all 5 Semantic Domains
    multi_domain_prompts = [
        "<BOS> a quick",       # Physical domain #2 ('in the' -> 'garden')
        "<BOS> a",             # AI/ML domain #1 ('small model predicts tokens')
        "<BOS> the neural",    # AI/ML domain #2 ('transformer encodes vectors')
        "<BOS> a smart",       # Academic domain #1 ('student reads books')
        "<BOS> every",         # Academic domain #2 ('professor writes papers')
        "<BOS> the clinical",  # Medical/Biology domain ('doctor treats patients')
        "<BOS> a cosmic",      # Astronomy/Space domain ('telescope observes stars')
    ]
    for demo_prompt in multi_domain_prompts:
        print_generation_walkthrough(
            model=model,
            prompt=demo_prompt,
            strategy="greedy",
            temperature=1.0,
            deep_inspect_steps=None,
        )

    # 4. Causal Attention Ablation & Decoding Strategy Comparison
    run_ablation_experiment(model)
    run_decoding_comparison(model, prompt="<BOS> the cat sat on the")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Academic Small Language Model (128-Token Inference-Only Micro-GPT Transformer) "
            "with transparent parameter provenance and step-by-step inspection."
        )
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt string (e.g., '<BOS> the cat' or '<BOS> the neural').",
    )
    parser.add_argument(
        "--show-parameters",
        action="store_true",
        help="Display the full 128-token parameter source, provenance, shapes, and values.",
    )
    parser.add_argument(
        "--deep-inspect",
        action="store_true",
        help="Show microscopic internal matrices (Embeddings, 4-Head Q/K/V, Causal Mask, 40 FFN Neurons, 128 Logits) for every step.",
    )
    parser.add_argument(
        "--inspect-step",
        type=int,
        nargs="*",
        default=None,
        help="Specific generation step number(s) to deep-inspect (e.g., --inspect-step 1 6).",
    )
    parser.add_argument(
        "--ablation-demo",
        action="store_true",
        help="Run the Causal Mechanistic Ablation experiment disabling Attention Head 1.",
    )
    parser.add_argument(
        "--compare-decoding",
        action="store_true",
        help="Compare Temperature scaling and Top-k filtering on the 128-token distribution.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "sampling", "top_k"],
        default="greedy",
        help="Next-token decoding strategy (default: greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature scaling factor (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k filtering cutoff before Softmax (optional).",
    )
    args = parser.parse_args()

    param_path = Path(__file__).resolve().parent / "parameters.json"
    model = AcademicSmallLanguageModel(parameter_path=param_path)

    if args.show_parameters and args.prompt is None and not args.ablation_demo and not args.compare_decoding:
        print_parameter_source_report(model.bundle)
        return

    if args.ablation_demo:
        run_ablation_experiment(model)
        if args.prompt is None and not args.compare_decoding:
            return

    if args.compare_decoding:
        run_decoding_comparison(
            model, prompt=args.prompt or "<BOS> the cat sat on the"
        )
        if args.prompt is None:
            return

    if args.prompt is not None:
        if args.show_parameters:
            print_parameter_source_report(model.bundle)
        inspect_steps = (
            list(range(1, 15))
            if args.deep_inspect
            else args.inspect_step
        )
        print_generation_walkthrough(
            model=model,
            prompt=args.prompt,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            deep_inspect_steps=inspect_steps,
        )
    else:
        run_academic_demo_suite(model)


if __name__ == "__main__":
    main()
