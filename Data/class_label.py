"""
Data/class_label.py  –  Vocab and learnable class-label embedder
=================================================================

8 thorax disease classes (NIH ChestX-ray14 subset):
    Atelectasis, Cardiomegaly, Effusion, Infiltration,
    Mass, Nodule, Pneumonia, Pneumothorax

Vocab
-----
    CLASSES    : list[str]  – ordered class names
    VOCAB      : dict[str → int]  – class name → index (0-7)
    IDX2CLASS  : dict[int → str]  – index → class name
    NUM_CLASSES: int  – 8

ClassLabelEmbedder
------------------
    Input  : list[str]  – label strings, possibly multi-label e.g. "Pneumonia|Effusion"
    Process: split by "|" → look up each valid label → mean-pool embeddings → project
    Output : Tensor [B, 1, output_dim]  (same shape as BioBERTContextEncoder output)

    Unknown labels (e.g. "No Finding") are silently skipped.
    If no valid label is found for a sample, a learned <unk> token is used.

Usage:
    embedder = ClassLabelEmbedder(embed_dim=256, output_dim=512).to(device)

    label_strs = ["Pneumonia|Effusion", "Atelectasis", "No Finding"]
    ctx = embedder(label_strs)   # [3, 1, 512]
"""

import torch
import torch.nn as nn


# =============================================================================
# Vocabulary
# =============================================================================

CLASSES: list = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
]

VOCAB:      dict = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX2CLASS:  dict = {idx: cls for cls, idx in VOCAB.items()}
NUM_CLASSES: int = len(CLASSES)   # 8


# =============================================================================
# ClassLabelEmbedder
# =============================================================================

class ClassLabelEmbedder(nn.Module):
    """
    Learnable embedding table for the 8 thorax disease class labels.

    Multi-label handling (e.g. "Pneumonia|Effusion"):
        - Split by '|'
        - Embed each recognised label
        - Mean-pool → linear projection → [B, 1, output_dim]

    Unknown labels (e.g. "No Finding", "Consolidation") are skipped.
    If a sample has no recognisable label, a dedicated <unk> token is used.

    Args:
        embed_dim  : Dimension of the internal embedding table (default: 256)
        output_dim : Final output dimension; must match LDM context_dim (default: 512)
    """

    UNK = '<unk>'

    def __init__(self, embed_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.embed_dim  = embed_dim
        self.output_dim = output_dim

        # NUM_CLASSES slots + 1 <unk> slot
        self.embedding = nn.Embedding(NUM_CLASSES + 1, embed_dim)
        self.unk_idx   = NUM_CLASSES          # index 8

        self.proj = nn.Linear(embed_dim, output_dim)

        n_emb  = self.embedding.weight.numel()
        n_proj = sum(p.numel() for p in self.proj.parameters())
        print(f'[ClassLabelEmbedder] vocab={NUM_CLASSES} (+1 unk)  '
              f'embed_dim={embed_dim}  output_dim={output_dim}  '
              f'params: {n_emb + n_proj:,}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def device(self):
        return self.embedding.weight.device

    def _label_str_to_tensor(self, label_str: str) -> torch.Tensor:
        """
        "Pneumonia|Effusion" → mean embedding [embed_dim].
        Unknown tokens skipped; falls back to <unk> if nothing recognised.
        """
        parts   = [p.strip() for p in label_str.split('|')]
        indices = [VOCAB[p] for p in parts if p in VOCAB]
        if not indices:
            indices = [self.unk_idx]
        idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.embedding(idx_t).mean(dim=0)    # [embed_dim]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, label_strs: list) -> torch.Tensor:
        """
        Args:
            label_strs : list[str]  e.g. ["Pneumonia|Effusion", "Atelectasis"]
        Returns:
            Tensor [B, 1, output_dim]
        """
        embs = torch.stack(
            [self._label_str_to_tensor(s) for s in label_strs]
        )                                        # [B, embed_dim]
        out = self.proj(embs).unsqueeze(1)       # [B, 1, output_dim]
        return out


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    # ── Vocab inspection ──────────────────────────────────────────────────────
    print('=== Vocab ===')
    for name, idx in VOCAB.items():
        print(f'  [{idx}] {name}')
    print(f'  [{NUM_CLASSES}] <unk>  (unknown / No Finding / etc.)\n')

    # ── Embedder ──────────────────────────────────────────────────────────────
    embedder = ClassLabelEmbedder(embed_dim=256, output_dim=512).to(device)

    test_cases = [
        'Atelectasis',               # single label, in vocab
        'Pneumonia|Effusion',        # multi-label, both in vocab
        'Mass|Nodule|Cardiomegaly',  # 3 labels
        'No Finding',                # unknown → <unk>
        'Consolidation|Effusion',    # one unknown, one known
    ]

    print('\n=== Embedding test ===')
    ctx = embedder(test_cases)       # [5, 1, 512]
    print(f'Input  : {len(test_cases)} samples')
    print(f'Output : {ctx.shape}')   # [5, 1, 512]
    assert ctx.shape == (len(test_cases), 1, 512), 'Shape mismatch!'

    for label, vec in zip(test_cases, ctx):
        print(f'  {label!r:40s} → {vec.shape}  norm={vec.norm().item():.4f}')

    print('\nClassLabelEmbedder test passed!')
