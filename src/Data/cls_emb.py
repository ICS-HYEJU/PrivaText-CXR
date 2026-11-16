"""
Data/class_label.py : Vocab and sequence-based class-label embedder
======================================================================

Vocab (18 tokens = 4 special + 14 disease labels)
--------------------------------------------------
    PAD=0  BOS=1  EOS=2  SEP=3
    Atelectasis=4  Cardiomegaly=5  Effusion=6  Infiltration=7
    Mass=8  Nodule=9  Pneumonia=10  Pneumothorax=11
    Consolidation=12  Edema=13  Emphysema=14  Fibrosis=15
    Pleural Thickening=16  Hernia=17

Sequence format (per sample)
-----------------------------
    BOS  label_1  SEP  label_2  SEP  ...  label_n  EOS  [PAD ...]

    "Pneumonia|Effusion"  -> [BOS, Pneumonia, SEP, Effusion, EOS]
                          -> [1, 10, 3, 6, 2]
    "Atelectasis"         -> [BOS, Atelectasis, EOS]
                          -> [1, 4, 2]
    "No Finding"          -> [BOS, EOS]          (unknown labels skipped)
                          -> [1, 2]

    Sequences in a batch are right-padded with PAD to match max seq_len.

ClassLabelEmbedder
------------------
    Input  : list[str]  - NIH label strings (possibly multi-label)
    Output : Tensor [B, seq_len, output_dim]
             output_dim must match LDM UNet context_dim (default: 512)

Usage:
    embedder = ClassLabelEmbedder(embed_dim=256, output_dim=512).to(device)

    label_strs = ["Pneumonia|Effusion", "Atelectasis", "No Finding"]
    ctx = embedder(label_strs)   # [3, seq_len, 512]
"""

import torch
import torch.nn as nn


# =============================================================================
# Vocabulary
# =============================================================================

VOCAB: dict = {
    'PAD': 0, 'BOS': 1, 'EOS': 2, 'SEP': 3,
    'Atelectasis': 4, 'Cardiomegaly': 5, 'Effusion': 6,
    'Infiltration': 7, 'Mass': 8, 'Nodule': 9, 'Pneumonia': 10, 'Pneumothorax': 11,
    'Consolidation': 12, 'Edema': 13, 'Emphysema': 14, 'Fibrosis': 15,
    'Pleural Thickening': 16, 'Hernia': 17,
}

IDX2TOKEN:  dict = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE: int  = len(VOCAB)     # 18

DISEASE_LABELS: list = [k for k in VOCAB if k not in ('PAD', 'BOS', 'EOS', 'SEP')]


# =============================================================================
# Sequence helpers
# =============================================================================

def encode_label_str(label_str: str) -> list:
    """
    Convert a NIH multi-label string to a token-index sequence.

    Format: BOS  label_1  SEP  label_2  SEP  ...  label_n  EOS

    Unknown tokens (e.g. "No Finding") are silently skipped.
    If no valid label is found, returns [BOS, EOS].

    Args:
        label_str : str  e.g. "Pneumonia|Effusion"
    Returns:
        list[int]  e.g. [1, 10, 3, 6, 2]
    """
    parts = [p.strip() for p in label_str.split('|')]
    valid = [p for p in parts if p in VOCAB]

    tokens = [VOCAB['BOS']]
    for i, label in enumerate(valid):
        tokens.append(VOCAB[label])
        if i < len(valid) - 1:
            tokens.append(VOCAB['SEP'])
    tokens.append(VOCAB['EOS'])
    return tokens


def decode_token_ids(token_ids: list) -> str:
    """
    Reverse encode_label_str: token indices -> human-readable string.

    Args:
        token_ids : list[int]
    Returns:
        str  e.g. "[BOS] Pneumonia [SEP] Effusion [EOS]"
    """
    return ' '.join(IDX2TOKEN.get(i, f'<{i}>') for i in token_ids)


def pad_sequences(sequences: list, pad_idx: int = VOCAB['PAD']) -> tuple:
    """
    Right-pad a list of token-index sequences to the same length.

    Args:
        sequences : list[list[int]]
        pad_idx   : int  padding token index (default: VOCAB['PAD'] = 0)
    Returns:
        padded : LongTensor  [B, max_seq_len]
        mask   : BoolTensor  [B, max_seq_len]
                 True = real token, False = PAD
    """
    max_len = max(len(s) for s in sequences)
    padded, mask = [], []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append(s + [pad_idx] * pad_len)
        mask.append([True] * len(s) + [False] * pad_len)
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(mask,   dtype=torch.bool),
    )


# =============================================================================
# ClassLabelEmbedder
# =============================================================================

class ClassLabelEmbedder(nn.Module):
    """
    Sequence-based learnable embedder for NIH thorax disease class labels.

    Encodes each label string as a padded token sequence, embeds it, then
    projects to output_dim. Output is a full sequence (not mean-pooled),
    enabling cross-attention over individual tokens in the UNet.

    Pipeline:
        label_str -> encode_label_str() -> [BOS, l1, SEP, l2, ..., EOS, PAD...]
                  -> nn.Embedding       -> [B, seq_len, embed_dim]
                  -> nn.Linear          -> [B, seq_len, output_dim]

    PAD positions are zeroed out after projection so they do not inject
    signal into the cross-attention keys/values.

    Args:
        embed_dim  : Dimension of the token embedding table (default: 256)
        output_dim : Output sequence dim; must match UNet context_dim (default: 512)
    """

    def __init__(self, embed_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.embed_dim  = embed_dim
        self.output_dim = output_dim

        # padding_idx=0: PAD embedding is always zero vector
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim,
                                      padding_idx=VOCAB['PAD'])
        self.proj      = nn.Linear(embed_dim, output_dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(f'[ClassLabelEmbedder] vocab_size={VOCAB_SIZE}  '
              f'embed_dim={embed_dim}  output_dim={output_dim}  '
              f'params: {n_params:,}')

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, label_strs: list) -> torch.Tensor:
        """
        Args:
            label_strs : list[str]  e.g. ["Pneumonia|Effusion", "Atelectasis"]
        Returns:
            Tensor [B, seq_len, output_dim]
            seq_len = max sequence length in the batch (shorter seqs PAD-padded)
            PAD positions are zero-filled in the output.
        """
        sequences       = [encode_label_str(s) for s in label_strs]
        padded, pad_mask = pad_sequences(sequences, pad_idx=VOCAB['PAD'])
        padded           = padded.to(self.device)     # [B, seq_len]

        emb = self.embedding(padded)                  # [B, seq_len, embed_dim]
        out = self.proj(emb)                          # [B, seq_len, output_dim]

        # Zero out PAD positions (proj bias would otherwise give non-zero output)
        real_mask = pad_mask.to(self.device).unsqueeze(-1)  # [B, seq_len, 1]
        out = out * real_mask                               # [B, seq_len, output_dim]

        return out


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}\n')

    # Vocab inspection
    print('=== Vocab ===')
    for token, idx in VOCAB.items():
        print(f'  [{idx:2d}] {token}')

    # Sequence encoding
    print('\n=== encode_label_str ===')
    test_strs = [
        'Pneumonia|Effusion',            # two known labels
        'Atelectasis',                   # single known label
        'Mass|Nodule|Cardiomegaly',      # three known labels
        'No Finding',                    # all unknown ¡æ [BOS, EOS]
        'Consolidation|Pleural Thickening',
    ]
    for s in test_strs:
        ids = encode_label_str(s)

    # Padding
    print('\n=== pad_sequences ===')
    seqs = [encode_label_str(s) for s in test_strs]
    padded, mask = pad_sequences(seqs)
    print(f'  padded shape : {padded.shape}')   # [5, max_seq_len]
    print(f'  mask   shape : {mask.shape}')
    print(f'  padded:\n{padded}')

    print('\n=== ClassLabelEmbedder ===')
    # Embedder
    embedder = ClassLabelEmbedder(embed_dim=256, output_dim=512).to(device)

    ctx = embedder(test_strs)   # [5, seq_len, 512]
    print(f'\n  input  : {len(test_strs)} samples')
    print(f'  output : {ctx.shape}')

    for s, vec in zip(test_strs, ctx):
        print(f'  {s!r:40s} ¡æ seq_len={vec.shape[0]}  '
              f'non-zero rows={int((vec.abs().sum(-1) > 0).sum())}')

    # PAD positions must be all-zero
    seqs_raw  = [encode_label_str(s) for s in test_strs]
    padded, _ = pad_sequences(seqs_raw)
    for b in range(len(test_strs)):
        seq_len = len(seqs_raw[b])
        pad_out = ctx[b, seq_len:, :]    # PAD region
        assert pad_out.abs().max().item() == 0.0, \
            f'Sample {b}: PAD positions are not zero!'
    print('\n  PAD positions are zero: OK')
    print('\nClassLabelEmbedder test passed!')