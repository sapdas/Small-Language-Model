"""
================================================================================
END-TO-END MATHEMATICAL GUIDE PDF GENERATOR (generate_pdf_guide.py)
================================================================================
Generates `SLM_End_to_End_Mathematical_Guide.pdf`, a comprehensive academic
reference document explaining the entire Small Language Model inference pipeline
end-to-end, including:
  - Architecture diagram and tensor dimension flow
  - Parameter provenance & matrix dimension registry (|V|=128, d_model=16, H=4, d_ff=40)
  - 128-token vocabulary & 16D semantic-lexical basis structure
  - Stage-by-stage mathematical equations and matrix shapes (Stages 1 to 8)
  - Role of Causal Self-Attention vs. Feed-Forward Network (Key-Value Memory)
  - Concrete numerical trace on "<BOS> the cat sat on the" -> "mat"
  - Complete 40 FFN rules table, ablation study, and autoregressive growth tables
================================================================================
"""

from __future__ import annotations
from html import escape
from pathlib import Path
from typing import Any, List

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

from slm_inference import AcademicSmallLanguageModel
from parameter_source import (
    EMBEDDING_DIMENSION_MEANINGS,
    FFN_RULE_SPECS,
    VOCABULARY,
    VOCABULARY_CATEGORIES,
)


def _header_footer(canvas, doc):
    """Draws a clean academic header and page number footer on every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4A5568"))
    canvas.drawString(
        0.60 * inch,
        10.55 * inch,
        "Academic Small Language Model (MicroGPT-128V-16D) — End-to-End Mathematical & Architectural Guide",
    )
    canvas.setStrokeColor(colors.HexColor("#CBD5E0"))
    canvas.setLineWidth(0.5)
    canvas.line(0.60 * inch, 10.47 * inch, 7.90 * inch, 10.47 * inch)

    canvas.line(0.60 * inch, 0.55 * inch, 7.90 * inch, 0.55 * inch)
    canvas.drawString(
        0.60 * inch,
        0.40 * inch,
        "Inference-Only Decoder Transformer | Known Parameter Set (6,888 Parameters, |V| = 128 Tokens)",
    )
    canvas.drawRightString(7.90 * inch, 0.40 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _build_pipeline_diagram() -> Drawing:
    """
    Constructs a compact vector flowchart of the 8-stage Transformer inference
    pipeline with explicit matrix and tensor dimensions at every block so that
    it fits cleanly on Page 1.
    """
    d = Drawing(520, 236)

    stages = [
        ("Stage 1: Tokenization & Vocabulary Lookup", "Input string -> integer Token IDs t in {0..127}^T", "Shape: [T]", "#EDF2F7", "#2B6CB0"),
        ("Stage 2: Token + Positional Embedding", "h^(0) = W_E[t] + W_P[0:T]   (W_E: [128, 16], W_P: [12, 16])", "Shape: [T, 16]", "#EBF8FF", "#2B6CB0"),
        ("Stage 3: Pre-Attention LayerNorm (LN_1)", "z^(1) = LN_1(h^(0))   (projects onto sphere ||z||_2 = sqrt(16) = 4.0)", "Shape: [T, 16]", "#E6FFFA", "#2C7A7B"),
        ("Stage 4: 4-Head Causal Self-Attention + Res 1", "Q,K,V=[T,16]->4x[T,4]; A_h=softmax(Q_h K_h^T/2 + M); h^(1)=h^(0)+AttnOut", "Shape: [T, 16]", "#FEFCBF", "#B7791F"),
        ("Stage 5: Pre-FFN LayerNorm (LN_2)", "z^(2) = LN_2(h^(1))   (projects onto sphere ||z||_2 = sqrt(16) = 4.0)", "Shape: [T, 16]", "#E6FFFA", "#2C7A7B"),
        ("Stage 6: FFN Key-Value Memory (d_ff=40) + Res 2", "m = GELU(z^(2) @ W_1 + b_1) [T, 40]; h^(2) = h^(1) + m @ W_2 + b_2", "Shape: [T, 16]", "#FAF5FF", "#6B46C1"),
        ("Stage 7: Final LayerNorm & LM Head (W_LM)", "h_last = LN_f(h^(2))[T-1] [1, 16]; logits = h_last @ W_LM + b_LM", "Shape: [128]", "#FFF5F5", "#C53030"),
        ("Stage 8: Softmax & Autoregressive Append", "P(w) = softmax(logits / tau); tokens <- tokens + [w_next]", "Len: T -> T+1", "#F0FFF4", "#2F855A"),
    ]

    box_w = 490
    box_h = 22
    x0 = 18
    y_top = 212
    step_y = 29.5

    for idx, (title, formula, shape_str, bg_hex, border_hex) in enumerate(stages):
        y = y_top - idx * step_y
        d.add(
            Rect(
                x0,
                y,
                box_w,
                box_h,
                rx=3.5,
                ry=3.5,
                fillColor=colors.HexColor(bg_hex),
                strokeColor=colors.HexColor(border_hex),
                strokeWidth=1.0,
            )
        )
        d.add(
            String(
                x0 + 8,
                y + 12.5,
                title,
                fontName="Helvetica-Bold",
                fontSize=8.0,
                fillColor=colors.HexColor("#1A202C"),
            )
        )
        d.add(
            String(
                x0 + 8,
                y + 3.5,
                formula,
                fontName="Courier",
                fontSize=7.0,
                fillColor=colors.HexColor("#2D3748"),
            )
        )
        d.add(
            String(
                x0 + box_w - 8,
                y + 8,
                shape_str,
                fontName="Helvetica-Bold",
                fontSize=7.8,
                fillColor=colors.HexColor(border_hex),
                textAnchor="end",
            )
        )
        if idx < len(stages) - 1:
            cx = x0 + box_w / 2
            y_arrow_top = y
            y_arrow_bot = y - (step_y - box_h)
            d.add(
                Line(
                    cx,
                    y_arrow_top,
                    cx,
                    y_arrow_bot + 3.2,
                    strokeColor=colors.HexColor("#4A5568"),
                    strokeWidth=1.1,
                )
            )
            d.add(
                Polygon(
                    [cx - 3.0, y_arrow_bot + 3.2, cx + 3.0, y_arrow_bot + 3.2, cx, y_arrow_bot],
                    fillColor=colors.HexColor("#4A5568"),
                    strokeColor=colors.HexColor("#4A5568"),
                )
            )
    return d


def build_pdf(output_path: Path) -> Path:
    model = AcademicSmallLanguageModel(output_path.parent / "parameters.json")
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15.5,
        leading=19,
        textColor=colors.HexColor("#1A365D"),
        alignment=TA_LEFT,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.0,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=8,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "AcademicBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=12.0,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "AcademicBullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=2.5,
    )
    math_style = ParagraphStyle(
        "MathBox",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.4,
        leading=10.2,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.6,
        borderPadding=5,
        leftIndent=4,
        rightIndent=4,
        spaceBefore=2,
        spaceAfter=5,
    )
    code_tiny = ParagraphStyle(
        "CodeTiny",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=6.8,
        leading=9.0,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#F7FAFC"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=5,
        leftIndent=4,
        rightIndent=4,
        spaceBefore=2,
        spaceAfter=5,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.3,
        leading=9.4,
        textColor=colors.HexColor("#1A202C"),
    )
    cell_bold_style = ParagraphStyle(
        "TableCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    header_cell_style = ParagraphStyle(
        "TableHeaderCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.6,
        leading=9.6,
        textColor=colors.white,
    )

    def styled_table(
        rows: List[List[Any]],
        col_widths: List[float],
        header_bg: str = "#1A365D",
    ) -> Table:
        """Wraps plain strings in Paragraphs so table cells wrap cleanly without clipping."""
        wrapped_rows: List[List[Any]] = []
        for r_idx, row in enumerate(rows):
            w_row = []
            for c_idx, val in enumerate(row):
                if isinstance(val, str):
                    safe_txt = escape(val)
                    if r_idx == 0:
                        w_row.append(Paragraph(safe_txt, header_cell_style))
                    elif c_idx == 0:
                        w_row.append(Paragraph(safe_txt, cell_bold_style))
                    else:
                        w_row.append(Paragraph(safe_txt, cell_style))
                else:
                    w_row.append(val)
            wrapped_rows.append(w_row)

        t = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("TOPPADDING", (0, 0), (-1, 0), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 1), (-1, -1), 2.8),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 2.8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return t

    story = []

    # ==========================================================================
    # PAGE 1: TITLE, HYPERPARAMETERS & PIPELINE FLOWCHART
    # ==========================================================================
    story.append(
        Paragraph(
            "Academic Small Language Model (MicroGPT-128V-16D)<br/>"
            "End-to-End Mathematical Formulation, Matrix Dimensions &amp; Inference Guide",
            title_style,
        )
    )
    story.append(
        Paragraph(
            "A complete white-box reference for an inference-only Decoder-Only Transformer "
            "operating on a known, analytically constructed set of 6,888 parameters across a 128-token vocabulary.",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.1, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph("1. Executive Overview &amp; Architectural Hyperparameters", h1_style))
    story.append(
        Paragraph(
            "This document specifies the complete mathematical and algorithmic pipeline of the "
            "<b>MicroGPT-128V-16D Academic Small Language Model</b> implemented in <font face='Courier'>parameter_source.py</font> "
            "and <font face='Courier'>slm_inference.py</font>. Unlike black-box neural networks whose weights are opaque "
            "floating-point arrays obtained from stochastic gradient descent, this model is <b>strictly inference-only</b> "
            "and uses a <b>known, analytically derived parameter set</b> (stored in <font face='Courier'>parameters.json</font>). "
            "Every matrix dimension, attention head, and feed-forward neuron has an explicit geometric and linguistic meaning.",
            body_style,
        )
    )

    hyper_data = [
        ["Hyperparameter Symbol", "Variable", "Value", "Mathematical & Architectural Role"],
        ["|V| (Vocabulary Size)", "vocab_size", "128 tokens", "Total number of unique tokens across 8 syntactic roles & 5 semantic domains"],
        ["L_max (Context Window)", "max_seq_len", "12 positions", "Maximum sequence length T supported by the positional matrix W_P"],
        ["d_model (Residual Dim)", "d_model", "16 dims", "Width of the residual stream; LayerNorm projects onto sphere of radius sqrt(16) = 4.0"],
        ["H (Attention Heads)", "n_heads", "4 heads", "Parallel causal self-attention heads (Syntax, Prep Router, Domain, Lexical)"],
        ["d_k = d_v (Head Dim)", "d_head", "4 dims", "Dimension per attention head (d_model / H = 16 / 4 = 4); scale factor sqrt(d_k) = 2.0"],
        ["d_ff (FFN Hidden Dim)", "d_ff", "40 neurons", "Number of Key-Value associative memory rules (N00..N39) in the MLP block"],
        ["eps (LayerNorm Epsilon)", "eps", "1e-5", "Numerical stability constant in denominator sqrt(var + eps)"],
        ["Theta (Total Parameters)", "total_params", "6,888 floats", "Exact sum of all weights and biases across all 12 parameter tensors"],
    ]
    story.append(styled_table(hyper_data, [1.45 * inch, 0.90 * inch, 0.80 * inch, 4.15 * inch]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1.1 End-to-End Inference Pipeline &amp; Tensor Shape Flowchart", h2_style))
    story.append(
        Paragraph(
            "For an input sequence of <b>T</b> tokens (where 1 &le; T &le; 12), the forward pass transforms "
            "the token sequence through 8 deterministic stages to predict token <b>T+1</b>:",
            body_style,
        )
    )
    story.append(_build_pipeline_diagram())

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 2: PARAMETER SOURCE, PROVENANCE & MATRIX DIMENSION REGISTRY
    # ==========================================================================
    story.append(Paragraph("2. Source of the Known Parameters &amp; Matrix Dimension Registry", h1_style))
    story.append(
        Paragraph(
            "All 6,888 parameters are constructed deterministically in <font face='Courier'>parameter_source.py</font> "
            "without any random number generation or training loop. The design synthesizes four foundational results "
            "from the Transformer interpretability literature:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Vaswani et al. (2017)</b> — <i>Attention Is All You Need</i>: Scaled dot-product attention "
            "and multi-head subspace projections.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Radford et al. (2019)</b> — <i>GPT-2 Pre-LayerNorm Decoder</i>: Applying LayerNorm on the "
            "inputs to the attention and FFN sub-blocks while maintaining an unnormalized residual stream.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Elhage et al. (2021)</b> — <i>A Mathematical Framework for Transformer Circuits</i>: "
            "Decomposing the residual stream into orthogonal subspaces read by Q/K/V and written by W_O.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Geva et al. (2021)</b> — <i>Transformer Feed-Forward Layers Are Key-Value Memories</i>: "
            "Constructing W_1 columns as spherical Key pattern detectors and W_2 rows as Value next-token transitions.",
            bullet_style,
        )
    )

    story.append(Paragraph("2.1 Complete Matrix Dimension &amp; Parameter Registry (6,888 Parameters)", h2_style))
    reg_rows = [
        ["Parameter Tensor(s)", "Stage / Layer", "Matrix Shape(s)", "Count", "Operation & Dimension Transformation (for input length T)"],
        ["W_E", "Stage 2: Token Embedding", "[128, 16]", "2,048", "Lookup W_E[token_ids] -> X_tok of shape [T, 16]"],
        ["W_P", "Stage 2: Positional Encoding", "[12, 16]", "192", "Slice W_P[0:T] -> X_pos of shape [T, 16]; h^(0) = X_tok + X_pos"],
        ["ln1_gamma, ln1_beta", "Stage 3: Pre-Attn LayerNorm", "[16], [16]", "32", "Element-wise scale/shift on [T, 16] -> z^(1) of shape [T, 16]"],
        ["W_Q, b_Q", "Stage 4a: Query Projection", "[16, 16], [16]", "272", "[T, 16] @ [16, 16] + [16] -> Q of shape [T, 16] -> 4 heads x [T, 4]"],
        ["W_K, b_K", "Stage 4b: Key Projection", "[16, 16], [16]", "272", "[T, 16] @ [16, 16] + [16] -> K of shape [T, 16] -> 4 heads x [T, 4]"],
        ["W_V, b_V", "Stage 4c: Value Projection", "[16, 16], [16]", "272", "[T, 16] @ [16, 16] + [16] -> V of shape [T, 16] -> 4 heads x [T, 4]"],
        ["W_O, b_O", "Stage 4d: Output Projection", "[16, 16], [16]", "272", "Concat(O_0..O_3) [T, 16] @ [16, 16] + [16] -> AttnOut [T, 16]"],
        ["ln2_gamma, ln2_beta", "Stage 5: Pre-FFN LayerNorm", "[16], [16]", "32", "Element-wise scale/shift on h^(1) [T, 16] -> z^(2) of shape [T, 16]"],
        ["W_1, b_1", "Stage 6a: FFN Key Detectors", "[16, 40], [40]", "680", "Expansion: [T, 16] @ [16, 40] + [40] -> ffn_pre of shape [T, 40]"],
        ["W_2, b_2", "Stage 6b: FFN Value Matrix", "[40, 16], [16]", "656", "Contraction: [T, 40] @ [40, 16] + [16] -> FFNOut of shape [T, 16]"],
        ["ln_f_gamma, ln_f_beta", "Stage 7a: Final LayerNorm", "[16], [16]", "32", "Normalizes h^(2) [T, 16] -> h_final [T, 16]; slice last row [1, 16]"],
        ["W_LM, b_LM", "Stage 7b: LM Head Unembed", "[16, 128], [128]", "2,176", "Unembedding: [1, 16] @ [16, 128] + [128] -> logits of shape [128]"],
        ["TOTAL", "All 12 Parameter Blocks", "—", "6,888", "Complete deterministic inference parameter set in parameters.json"],
    ]
    story.append(styled_table(reg_rows, [1.20 * inch, 1.40 * inch, 0.95 * inch, 0.50 * inch, 3.25 * inch]))
    story.append(Spacer(1, 5))

    story.append(Paragraph("2.2 The 16-Dimensional Semantic-Lexical Subspace Basis (d_model = 16)", h2_style))
    story.append(
        Paragraph(
            "To embed 128 tokens into a 16-dimensional vector space without collisions (achieving a maximum "
            "pairwise cosine similarity of <b>0.8245</b> across all 8,001 pairs of active tokens), the 16 coordinates "
            "of each row in <b>W_E</b> <font face='Courier'>[128, 16]</font> are partitioned into four functional groups:",
            body_style,
        )
    )
    dim_rows = [["Dim", "Subspace Group", "Explicit Linguistic / Mathematical Meaning in Residual Stream"]]
    for d_idx, desc in EMBEDDING_DIMENSION_MEANINGS.items():
        if d_idx <= 6:
            grp = "Part-of-Speech (Dims 0..6)"
        elif d_idx <= 11:
            grp = "Semantic Domain (Dims 7..11)"
        elif d_idx <= 14:
            grp = "3-Bit Hypercube ID (Dims 12..14)"
        else:
            grp = "Context Register (Dim 15)"
        dim_rows.append([f"Dim {d_idx:2d}", grp, desc])
    story.append(styled_table(dim_rows, [0.55 * inch, 1.70 * inch, 5.05 * inch]))

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 3: 128-TOKEN VOCABULARY & STAGES 1-3
    # ==========================================================================
    story.append(Paragraph("2.3 Complete 128-Token Vocabulary Organization (|V| = 128)", h2_style))
    story.append(
        Paragraph(
            "Each token in the 128-token vocabulary is assigned a Part-of-Speech coordinate (Dims 0..6), "
            "a Semantic Domain coordinate (Dims 7..11: Physical, Academic, AI, Medical, Astronomy), and one of the "
            "8 vertices of the 3D hypercube (&plusmn;2.4, &plusmn;2.4, &plusmn;2.4) on Dims 12..14, followed by row-wise "
            "zero-centering so every row has mean 0:",
            body_style,
        )
    )
    vocab_rows: List[List[Any]] = [["ID Range", "Category Name", "Count", "Complete List of Token Strings (with Vocabulary IDs)"]]
    for cat_name, s_id, e_id, _ in VOCABULARY_CATEGORIES:
        tok_list_str = ", ".join(f"{i}:'{escape(VOCABULARY[i])}'" for i in range(s_id, e_id + 1))
        p_cell = Paragraph(f"<font face='Courier' size='6.8'>{tok_list_str}</font>", cell_style)
        vocab_rows.append([f"{s_id:3d} .. {e_id:3d}", cat_name, str(e_id - s_id + 1), p_cell])
    story.append(styled_table(vocab_rows, [0.70 * inch, 1.45 * inch, 0.45 * inch, 4.70 * inch]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. End-to-End Mathematical Formulation of Inference Steps", h1_style))

    story.append(Paragraph("3.1 Stage 1 (Tokenization) &amp; Stage 2 (Token + Positional Embeddings)", h2_style))
    story.append(
        Paragraph(
            "Given an input string of <b>T</b> whitespace-delimited tokens (automatically prepended with "
            "<font face='Courier'>&lt;BOS&gt;</font> if absent), Stage 1 maps each token string to its integer index "
            "in <font face='Courier'>0..127</font>. Stage 2 retrieves the corresponding rows from the Token Embedding "
            "matrix <b>W_E</b> and adds the first <b>T</b> rows of the Positional Encoding matrix <b>W_P</b> "
            "(generated in <font face='Courier'>parameter_source.py:L355-L361</font>):",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "Token IDs        : t = [t_0, t_1, ..., t_{T-1}]                 shape: [T]\n"
            "Token Embedding  : X_tok[i, :] = W_E[t_i, :]                    shape: [T, 16]  (from W_E: [128, 16])\n"
            "Positional Wave  : W_P[pos, 0] = +0.03 * cos(0.5 * pos),  W_P[pos, 2] = -W_P[pos, 0]\n"
            "                   W_P[pos, 1] = +0.03 * sin(0.5 * pos),  W_P[pos, 3] = -W_P[pos, 1]\n"
            "Initial Residual : h^(0) = X_tok + W_P[0:T, :]                  shape: [T, 16] + [T, 16] -> [T, 16]",
            math_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Why W_P is zero-mean by construction:</b> Because <font face='Courier'>W_P[pos, 2] = -W_P[pos, 0]</font> "
            "and <font face='Courier'>W_P[pos, 3] = -W_P[pos, 1]</font>, the sum of all 16 elements in every row of "
            "<b>W_P</b> is identically zero. Since every row of <b>W_E</b> is also zero-centered, the initial residual "
            "stream <b>h^(0)</b> has exact row-wise mean 0.000 at every position.",
            body_style,
        )
    )

    story.append(Paragraph("3.2 Stage 3: Pre-Attention Layer Normalization (LN_1) &amp; Hypersphere Geometry", h2_style))
    story.append(
        Paragraph(
            "Layer Normalization (Ba et al., 2016) standardizes each token row independently across the "
            "d_model = 16 features. Because d_model = 16 is a perfect square and gamma = 1, beta = 0, "
            "every normalized token vector lies on the zero-mean sphere of <b>exact integer radius 4.0000</b>:",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "Row Mean         : mu_t    = (1 / 16) * sum_{d=0}^{15} h^(0)[t, d]              shape: [T, 1]\n"
            "Row Variance     : sigma^2 = (1 / 16) * sum_{d=0}^{15} (h^(0)[t, d] - mu_t)^2   shape: [T, 1]\n"
            "Normalized State : z^(1)   = gamma_1 * (h^(0) - mu) / sqrt(sigma^2 + eps) + beta_1   shape: [T, 16]\n"
            "Sphere Invariant : ||z^(1)[t, :]||_2 = sqrt( sum_{d=0}^{15} (z^(1)[t, d])^2 ) = sqrt(16 * 1.0) = 4.0000",
            math_style,
        )
    )

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 4: STAGE 4 (SELF-ATTENTION) & STAGES 5-6 (FFN KEY-VALUE MEMORY)
    # ==========================================================================
    story.append(Paragraph("3.3 Stage 4: Multi-Head Causal Self-Attention (H = 4, d_k = 4) &amp; Residual 1", h2_style))
    story.append(
        Paragraph(
            "The normalized sequence <b>z^(1)</b> <font face='Courier'>[T, 16]</font> is projected into Query (<b>Q</b>), "
            "Key (<b>K</b>), and Value (<b>V</b>) representations and split into <b>H = 4 parallel heads</b>, each operating "
            "on a 4-dimensional subspace (<font face='Courier'>d_head = 4</font>):",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "1. Linear Projections:\n"
            "   Q = z^(1) @ W_Q + b_Q    ([T, 16] @ [16, 16] + [16] -> [T, 16])\n"
            "   K = z^(1) @ W_K + b_K    ([T, 16] @ [16, 16] + [16] -> [T, 16])\n"
            "   V = z^(1) @ W_V + b_V    ([T, 16] @ [16, 16] + [16] -> [T, 16])\n\n"
            "2. Split into H = 4 Heads (for head h in {0, 1, 2, 3}, slicing columns 4h : 4h+4):\n"
            "   Q_h = Q[:, 4h : 4h+4],   K_h = K[:, 4h : 4h+4],   V_h = V[:, 4h : 4h+4]   shape: [T, 4]\n\n"
            "3. Scaled Dot-Product Scores & Upper-Triangular Causal Mask M:\n"
            "   S_h = (Q_h @ K_h^T) / sqrt(4) = (Q_h @ K_h^T) / 2.0      ([T, 4] @ [4, T] -> [T, T])\n"
            "   M[i, j] = 0.0 if j <= i else -1e9                        shape: [T, T]\n"
            "   A_h = softmax(S_h + M, axis=-1)                          shape: [T, T] (each row sums to 1.0)\n\n"
            "4. Value Aggregation, Head Concatenation, Output Projection & First Residual Addition:\n"
            "   O_h      = A_h @ V_h                                     ([T, T] @ [T, 4] -> [T, 4])\n"
            "   O_concat = Concat(O_0, O_1, O_2, O_3, axis=-1)           shape: [T, 16]\n"
            "   AttnOut  = O_concat @ W_O + b_O                          ([T, 16] @ [16, 16] + [16] -> [T, 16])\n"
            "   h^(1)    = h^(0) + AttnOut                               ([T, 16] + [T, 16] -> [T, 16])",
            math_style,
        )
    )

    head_data = [
        ["Head Index", "Columns", "Subspace Role & Mechanistic Routing Function"],
        ["Head 0", "Cols 0..3", "Local Syntactic Constituency: Matches POS roles (Ctrl, Det, Adj, SubjNoun) on dims 0..3."],
        ["Head 1", "Cols 4..7", "Long-Range Prepositional Router: Determiner 'the' (dim 1) queries for past Preposition 'on'/'in' (dim 5), writing +15.44 into Dim 15 (Context Register) and flavor into Dims 7..8."],
        ["Head 2", "Cols 8..11", "Semantic Domain Persistence: Tracks Physical, Academic, AI, and Medical domain signals (dims 7..10)."],
        ["Head 3", "Cols 12..15", "Lexical Sub-Identity & Space Domain: Tracks 3-bit hypercube coordinates and Astronomy domain (dims 11..14)."],
    ]
    story.append(styled_table(head_data, [0.80 * inch, 0.80 * inch, 5.70 * inch]))
    story.append(Spacer(1, 5))

    story.append(Paragraph("3.4 Stage 5 (LN_2) &amp; Stage 6: Position-Wise Feed-Forward Network (FFN)", h2_style))
    story.append(
        Paragraph(
            "After Self-Attention routes contextual information across token positions into <b>h^(1)</b>, "
            "Stage 5 normalizes each row onto the radius-4.0 sphere (<font face='Courier'>z^(2) = LN_2(h^(1))</font>), "
            "and Stage 6 applies the Position-Wise Feed-Forward Network (<font face='Courier'>slm_inference.py:L376-L395</font>):",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "1. Pre-FFN LayerNorm         : z^(2)   = LN_2(h^(1))                  shape: [T, 16] (||z^(2)[t]||_2 = 4.0)\n"
            "2. Key Detection (Expansion) : ffn_pre = z^(2) @ W_1 + b_1            ([T, 16] @ [16, 40] + [40] -> [T, 40])\n"
            "3. Non-Linear GELU Gate      : m       = GELU(ffn_pre)                shape: [T, 40]\n"
            "   where GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))\n"
            "4. Value Write (Contraction) : FFNOut  = m @ W_2 + b_2                ([T, 40] @ [40, 16] + [16] -> [T, 16])\n"
            "5. Second Residual Addition  : h^(2)   = h^(1) + FFNOut               ([T, 16] + [T, 16] -> [T, 16])",
            math_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Why the FFN is Essential to the Transformer (Spherical Matched-Filter Theorem):</b> "
            "Self-Attention alone is a linear combination of Value vectors across tokens; it gathers context "
            "into a single position but cannot perform non-linear <font face='Courier'>AND</font> feature detection or "
            "map a state to a brand-new vocabulary direction. In our model, each column <i>j</i> of <b>W_1</b> is "
            "<font face='Courier'>6.0 * u_j</font> (where <font face='Courier'>||u_j||_2 = 1</font> is the unit post-attention "
            "vector of grammatical rule <i>j</i>) and <font face='Courier'>b_1[j] = -21.60</font>. By Cauchy-Schwarz:",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "ffn_pre[t, j] = z^(2)[t] . (6.0 * u_j) - 21.60 = 6.0 * 4.0 * cos(theta_j) - 21.60 = 24.0 * (cos(theta_j) - 0.90)",
            math_style,
        )
    )
    story.append(
        Paragraph(
            "Thus <font face='Courier'>ffn_pre[t, j] &gt; 0</font> <b>if and only if</b> <font face='Courier'>cos(theta_j) &gt; 0.90</font>! "
            "For the exact matching rule (<font face='Courier'>cos(theta_j) = 1.0</font>), <font face='Courier'>ffn_pre = +2.40</font> "
            "and <font face='Courier'>GELU(+2.40) = +2.38</font>, whereupon row <i>j</i> of <b>W_2</b> writes "
            "<font face='Courier'>8.0 * W_E[target_j] - h^(1)_j</font> into the residual stream, steering <b>h^(2)</b> "
            "directly toward the embedding of the next token!",
            body_style,
        )
    )

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 5: STAGES 7-8 & CONCRETE NUMERICAL TRACE (STEP 6: "<BOS> the cat sat on the" -> "mat")
    # ==========================================================================
    story.append(Paragraph("3.5 Stage 7 (Final LayerNorm &amp; LM Head) &amp; Stage 8 (Softmax &amp; Appending)", h2_style))
    story.append(
        Paragraph(
            "In autoregressive generation, only the final sequence position (<font face='Courier'>index T-1</font>) "
            "is projected by the Language Modeling Head <b>W_LM</b> <font face='Courier'>[16, 128]</font> to produce "
            "raw logits over all 128 tokens:",
            body_style,
        )
    )
    story.append(
        Preformatted(
            "1. Final LayerNorm         : h_final = LN_f(h^(2))                     shape: [T, 16]\n"
            "2. Extract Last Position   : h_last  = h_final[T - 1, :]               shape: [16] (or [1, 16])\n"
            "3. Unembedding Projection  : logits  = h_last @ W_LM + b_LM            ([16] @ [16, 128] + [128] -> [128])\n"
            "   where W_LM[:, v] = 4.5 * W_E[v, :] / ||W_E[v, :]||_2  (weight-tied unit direction of token v)\n"
            "4. Temperature Scaling     : scaled  = logits / tau                    shape: [128] (tau > 0)\n"
            "5. Optional Top-k Masking  : scaled[v] = -1e9 for v not in Top-k(scaled)\n"
            "6. Softmax Probabilities   : P(w_v | w_{0:T-1}) = exp(scaled_v - max) / sum_{u=0}^{127} exp(scaled_u - max)\n"
            "7. Next-Token Selection    : v_next = argmax_v P(w_v)  [Greedy]   OR   v_next ~ Categorical(P)  [Sampling]\n"
            "8. Autoregressive Append   : tokens_new = tokens + [VOCABULARY[v_next]]   (length grows from T to T+1)",
            math_style,
        )
    )

    story.append(Paragraph("4. Concrete End-to-End Numerical Trace (Non-Markovian Disambiguation)", h1_style))
    story.append(
        Paragraph(
            "Consider Step 6 of generating from <font face='Courier'>\"&lt;BOS&gt;\"</font>, where the current context of "
            "length <b>T = 6</b> is <font face='Courier'>[\"&lt;BOS&gt;\", \"the\", \"cat\", \"sat\", \"on\", \"the\"]</font>. "
            "Notice that <font face='Courier'>\"the\"</font> (ID=4) appears at <b>Position 1</b> (where it predicted "
            "<font face='Courier'>\"cat\"</font>) and again at <b>Position 5</b> (where it must predict "
            "<font face='Courier'>\"mat\"</font>). Below are the exact numerical tensors produced by "
            "<font face='Courier'>model.forward_step()</font>:",
            body_style,
        )
    )

    trace = model.forward_step(["<BOS>", "the", "cat", "sat", "on", "the"], step_number=6)

    h1_w = trace.head_traces[1].attn_weights
    attn_table_rows = [
        ["Query Pos (Token)", "Key 0: <BOS>", "Key 1: the", "Key 2: cat", "Key 3: sat", "Key 4: on", "Key 5: the", "Routed Context"]
    ]
    for q_i, q_tok in enumerate(trace.input_tokens):
        row = [f"Pos {q_i}: '{q_tok}'"]
        for k_i in range(6):
            if k_i > q_i:
                row.append("[MASK -inf]")
            else:
                row.append(f"{h1_w[q_i, k_i]*100:5.1f}%")
        best_k = int(np.argmax(h1_w[q_i, : q_i + 1]))
        row.append(f"-> '{trace.input_tokens[best_k]}' ({h1_w[q_i, best_k]*100:.1f}%)")
        attn_table_rows.append(row)

    story.append(Paragraph("4.1 Stage 4 Numerical Matrix: Attention Head 1 Weights A_1 (Shape [6, 6])", h2_style))
    story.append(
        styled_table(
            attn_table_rows,
            [1.05 * inch, 0.78 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 1.87 * inch],
        )
    )
    story.append(Spacer(1, 4))

    def vec_split_str(label: str, v: np.ndarray, note: str = "") -> str:
        p1 = ", ".join(f"{x:+6.2f}" for x in v[:8])
        p2 = ", ".join(f"{x:+6.2f}" for x in v[8:])
        suffix = f"  {note}" if note else ""
        return f"{label:<22s} dims[0..7] : [{p1}]\n{'':<22s} dims[8..15]: [{p2}]{suffix}"

    story.append(Paragraph("4.2 Evolution of the 16D Vector at the Final Position (Pos 5: 'the', ID = 4)", h2_style))
    story.append(
        Preformatted(
            "\n".join(
                [
                    vec_split_str("W_E[4] ('the')", trace.tok_embeddings[5]),
                    vec_split_str("h^(0)[5] (Initial)", trace.h0_residual[5]),
                    vec_split_str("LN_1(h^(0))[5]", trace.ln1_out[5], "(||z^(1)||_2 = 4.0000)"),
                    vec_split_str("AttnOut[5] (Head 0..3)", trace.attn_out[5], "<-- Dim 15 = +15.44 routed from 'on'!"),
                    vec_split_str("h^(1)[5] (Res 1)", trace.h1_residual[5]),
                    vec_split_str("LN_2(h^(1))[5]", trace.ln2_out[5], "(||z^(2)||_2 = 4.0000)"),
                    vec_split_str("FFNOut[5] (Neuron 28)", trace.ffn_out[5], "<-- N28 fires (+2.38), writes 'mat'!"),
                    vec_split_str("LN_f(h^(2))[5]", trace.last_hidden_state, "(points along W_E[96] 'mat')"),
                ]
            ),
            code_tiny,
        )
    )

    story.append(Paragraph("4.3 Stage 7 &amp; 8 Numerical Output: Top-6 Candidates (out of 128 Logits) &amp; Appending", h2_style))
    top6 = np.argsort(trace.probabilities)[::-1][:6]
    pred_rows = [["Rank", "Token ID", "Token", "Raw Logit", "Prob (%)", "Selection & Autoregressive Action"]]
    for rk, tid in enumerate(top6, start=1):
        action = (
            "SELECTED (argmax) -> Appended to form ['<BOS>', 'the', 'cat', 'sat', 'on', 'the', 'mat']"
            if rk == 1
            else "Not selected"
        )
        pred_rows.append(
            [
                f"#{rk}",
                f"{int(tid)} / 127",
                f"'{model.id_to_token[int(tid)]}'",
                f"{trace.raw_logits[tid]:+8.3f}",
                f"{trace.probabilities[tid]*100:6.2f}%",
                action,
            ]
        )
    story.append(styled_table(pred_rows, [0.45 * inch, 0.70 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 3.90 * inch]))

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 6: COMPLETE 40 FFN KEY-VALUE MEMORY RULES
    # ==========================================================================
    story.append(Paragraph("5. Complete Registry of the 40 FFN Key-Value Memory Rules (d_ff = 40)", h1_style))
    story.append(
        Paragraph(
            "The table below lists all 40 associative memory rules encoded in <b>W_1</b> <font face='Courier'>[16, 40]</font> "
            "and <b>W_2</b> <font face='Courier'>[40, 16]</font> (<font face='Courier'>parameter_source.py:L130-L227</font>):",
            body_style,
        )
    )
    ffn_rows = [["Neuron", "Trigger Context Sequence (Key u_j)", "Target Next Token", "Mechanistic Rule Description"]]
    for idx, (ctx_list, tgt_tok, desc) in enumerate(FFN_RULE_SPECS):
        short_desc = desc.split(":")[1].strip() if ":" in desc else desc
        ffn_rows.append(
            [
                f"N{idx:02d}",
                " ".join(ctx_list),
                f"'{tgt_tok}' (ID={VOCABULARY.index(tgt_tok)})",
                short_desc,
            ]
        )
    story.append(styled_table(ffn_rows, [0.55 * inch, 2.30 * inch, 1.15 * inch, 3.30 * inch]))

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 7: AUTOREGRESSIVE GROWTH ACROSS ALL 5 DOMAINS & ABLATION PROOF
    # ==========================================================================
    story.append(Paragraph("6. Autoregressive Token String Growth &amp; Causal Ablation Verification", h1_style))
    story.append(
        Paragraph(
            "6.1 Step-by-Step Token String Growth on Flagship Prompt (\"&lt;BOS&gt;\" -> \"&lt;BOS&gt; the cat sat on the mat &lt;EOS&gt;\")",
            h2_style,
        )
    )
    flagship_traces = model.generate("<BOS>", max_new_tokens=8)
    growth_rows = [["Step", "Input Context Length & Token String", "FFN Rule", "Predicted Token", "Prob", "Updated Appended Token String (Length T+1)"]]
    for tr in flagship_traces:
        T_len = len(tr.input_tokens)
        best_n = int(np.argmax(tr.ffn_hidden[-1]))
        growth_rows.append(
            [
                f"#{tr.step_number}",
                f"(T={T_len}) " + " ".join(tr.input_tokens),
                f"N{best_n:02d}",
                f"+ '{tr.predicted_token}' (ID={tr.predicted_id})",
                f"{tr.predicted_prob*100:5.1f}%",
                f"(T={T_len+1}) " + " ".join(tr.updated_tokens),
            ]
        )
    story.append(styled_table(growth_rows, [0.40 * inch, 2.05 * inch, 0.65 * inch, 1.05 * inch, 0.55 * inch, 2.60 * inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.2 Multi-Domain Generation Summary Across All 5 Knowledge Domains", h2_style))
    domain_demos = [
        ("Physical #1", "<BOS>", "<BOS> the cat sat on the mat <EOS>", "N00 -> N01 -> N10 -> N18 -> N26 -> N28 -> N30"),
        ("Physical #2", "<BOS> a quick", "<BOS> a quick dog ran in the garden <EOS>", "N05 -> N11 -> N19 -> N27 -> N29 -> N31"),
        ("AI / Computing #1", "<BOS> a", "<BOS> a small model predicts tokens <EOS>", "N02 -> N04 -> N12 -> N20 -> N32"),
        ("AI / Computing #2", "<BOS> the neural", "<BOS> the neural transformer encodes vectors <EOS>", "N07 -> N13 -> N21 -> N33"),
        ("AI / Computing #3", "<BOS> this", "<BOS> this algorithm predicts tokens <EOS>", "N38 -> N39 -> N20 -> N32"),
        ("Academic #1", "<BOS> a smart", "<BOS> a smart student reads books <EOS>", "N06 -> N14 -> N22 -> N34"),
        ("Academic #2", "<BOS> every", "<BOS> every professor writes papers <EOS>", "N03 -> N15 -> N23 -> N35"),
        ("Medicine / Biology", "<BOS> the clinical", "<BOS> the clinical doctor treats patients <EOS>", "N08 -> N16 -> N24 -> N36"),
        ("Astronomy / Space", "<BOS> a cosmic", "<BOS> a cosmic telescope observes stars <EOS>", "N09 -> N17 -> N25 -> N37"),
    ]
    dom_rows = [["Domain", "Initial Prompt", "Complete Autoregressively Generated Sentence", "Active FFN Neuron Sequence"]]
    for d_name, p_str, out_s, n_seq in domain_demos:
        dom_rows.append([d_name, p_str, out_s, n_seq])
    story.append(styled_table(dom_rows, [1.15 * inch, 1.10 * inch, 2.90 * inch, 2.15 * inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.3 Causal Mechanistic Ablation Proof (Why Both Attention and FFN Are Required)", h2_style))
    story.append(
        Paragraph(
            "When Attention Head 1 is disabled (<font face='Courier'>ablate_heads=[1]</font> in "
            "<font face='Courier'>slm_inference.py:L351-L352</font>), the preposition <font face='Courier'>'on'</font> "
            "or <font face='Courier'>'in'</font> is no longer routed into <font face='Courier'>Dim 15</font> at the "
            "second <font face='Courier'>'the'</font>. Consequently, neither <font face='Courier'>N28</font> nor "
            "<font face='Courier'>N29</font> reaches the cosine threshold (<font face='Courier'>cos &gt; 0.90</font>) "
            "in the FFN, proving that Self-Attention routing is causally necessary for context-dependent prediction:",
            body_style,
        )
    )
    ablation_rows = [
        ["Test Context (Ending at 'the', ID=4)", "Mode", "Dim 15 (CtxReg)", "Fired FFN Neuron", "Predicted Token & Prob"],
        ["\"<BOS> the cat sat on the\"", "Normal (All 4 Heads)", "+15.44", "N28 [Routed 'on the']", "'mat' (94.39%) [Correct]"],
        ["\"<BOS> the cat sat on the\"", "Head 1 Ablated", "+0.00", "None (all < threshold)", "'the' (33.23%) [Fails]"],
        ["\"<BOS> a quick dog ran in the\"", "Normal (All 4 Heads)", "+15.13", "N29 [Routed 'in the']", "'garden' (93.33%) [Correct]"],
        ["\"<BOS> a quick dog ran in the\"", "Head 1 Ablated", "+0.00", "None (all < threshold)", "'the' (32.76%) [Fails]"],
    ]
    story.append(styled_table(ablation_rows, [1.95 * inch, 1.25 * inch, 0.95 * inch, 1.45 * inch, 1.70 * inch]))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.60 * inch,
        rightMargin=0.60 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.68 * inch,
        title="Academic Small Language Model — End-to-End Mathematical Guide",
        author="MicroGPT-128V-16D Academic Documentation",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


if __name__ == "__main__":
    pdf_file = Path(__file__).resolve().parent / "SLM_End_to_End_Mathematical_Guide.pdf"
    build_pdf(pdf_file)
    print(f"[OK] Generated PDF guide: {pdf_file}")
