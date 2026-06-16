"""
Diffusion/context_encoder.py  ?  BioBERT context encoder for LDM conditioning
==============================================================================

Encodes text conditions into embeddings for LDM cross-attention.
Output shape: [B, 1, output_dim]  (matches LDM context_dim)

Two modes:
    'label'       : Short class label  (e.g. "pneumonia", "normal")
                    ¡æ CLS token ¡æ project ¡æ [B, 1, output_dim]
    'description' : Clinical description (e.g. "Bilateral pleural effusion")
                    ¡æ mean-pool over tokens ¡æ project ¡æ [B, 1, output_dim]

Usage (before LDM forward):
    encoder = BioBERTContextEncoder(output_dim=512).to(device)

    # Case 1: class label
    labels = ["pneumonia", "normal"]
    c = encoder(labels, mode='label')          # [2, 1, 512]

    # Case 2: clinical description
    descs  = ["Bilateral pleural effusion ...", "No acute process."]
    c = encoder(descs,  mode='description')    # [2, 1, 512]

    # Pass to LDM
    loss, loss_dict = model.training_step({'image': x, 'context': c})
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=8, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim) # 1001, 512

    def forward(self, label):
        c = self.embedding(label)
        return c

class BioBERTContextEncoder(nn.Module):
    """
    BioBERT-based text encoder that produces fixed-size context embeddings.

    Args:
        model_name : HuggingFace model ID (default: dmis-lab/biobert-v1.1)
        output_dim : Output dim; must match LDM UNet context_dim (default: 512)
        max_length : Max token length for description mode (default: 128)
        freeze     : Freeze BioBERT weights (default: True)
    """

    BIOBERT_DIM = 768    # BioBERT-base hidden size

    def __init__(self,
                 model_name: str = 'dmis-lab/biobert-v1.1',
                 output_dim: int = 512,
                 max_length: int = 128,
                 freeze: bool = True):
        super().__init__()

        self.output_dim  = output_dim
        self.max_length  = max_length
        self._freeze     = freeze

        # ¦¡¦¡ Load BioBERT ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        print(f'[BioBERTContextEncoder] Loading {model_name} ...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert      = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            print(f'[BioBERTContextEncoder] BioBERT frozen')

        # ¦¡¦¡ Projection: BioBERT hidden ¡æ LDM context_dim ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        self.proj = nn.Linear(self.BIOBERT_DIM, output_dim)

        n_proj = sum(p.numel() for p in self.proj.parameters())
        print(f'[BioBERTContextEncoder] proj params: {n_proj:,}  '
              f'({self.BIOBERT_DIM} ¡æ {output_dim})')

    # ¦¡¦¡ Helpers ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize(self, texts: list, max_length: int) -> dict:
        return self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = max_length,
            return_tensors = 'pt',
        ).to(self.device)

    def _bert_forward(self, enc: dict):
        """Run BERT with no_grad when frozen."""
        if self._freeze:
            with torch.no_grad():
                return self.bert(**enc)
        return self.bert(**enc)

    # ¦¡¦¡ Encoding modes ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def encode_label(self, labels: list) -> torch.Tensor:
        """
        Encode short class labels via CLS token.

        Args:
            labels : list[str]  e.g. ["pneumonia", "normal"]
        Returns:
            [B, 1, output_dim]
        """
        enc = self._tokenize(labels, max_length=32)   # labels are short
        out = self._bert_forward(enc)

        cls = out.last_hidden_state[:, 0, :]           # [B, BIOBERT_DIM]  CLS
        emb = self.proj(cls).unsqueeze(1)              # [B, 1, output_dim]
        return emb

    def encode_description(self, descriptions: list) -> torch.Tensor:
        """
        Encode clinical descriptions via mean-pooling over non-padding tokens.

        Args:
            descriptions : list[str]  e.g. ["Bilateral pleural effusion ..."]
        Returns:
            [B, 1, output_dim]
        """
        enc  = self._tokenize(descriptions, max_length=self.max_length)
        out  = self._bert_forward(enc)

        mask = enc['attention_mask'].unsqueeze(-1).float()          # [B, seq, 1]
        mean = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # [B, BIOBERT_DIM]
        emb  = self.proj(mean).unsqueeze(1)                         # [B, 1, output_dim]
        return emb

    def forward(self, texts: list, mode: str = 'label') -> torch.Tensor:
        """
        Args:
            texts : list[str]
            mode  : 'label' | 'description'
        Returns:
            [B, 1, output_dim]
        """
        if mode == 'label':
            return self.encode_label(texts)
        elif mode == 'description':
            return self.encode_description(texts)
        else:
            raise ValueError(f"mode must be 'label' or 'description', got '{mode}'")


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    encoder = BioBERTContextEncoder(
        model_name = 'dmis-lab/biobert-base-cased-v1.1',
        output_dim = 512,
        freeze     = True,
    ).to(device)

    # Case 1: class label
    labels = [
        'pneumonia',
        'normal',
        'pleural effusion',
        'cardiomegaly',
    ]
    c_label = encoder(labels, mode='label')
    print(f'[label]       input : {labels}')
    print(f'[label]       output: {c_label.shape}')   # [4, 1, 512]

    # Case 2: clinical description
    descriptions = [
        'Bilateral pleural effusion with mild cardiomegaly.',
        'No acute cardiopulmonary process. Lungs are clear.',
        'Interstitial opacities in right lower lobe consistent with pneumonia.',
        'Mild pulmonary edema with bilateral hilar prominence.',
    ]
    c_desc = encoder(descriptions, mode='description')
    print(f'\n[description] input : {descriptions[0][:50]}...')
    print(f'[description] output: {c_desc.shape}')    # [4, 1, 512]

    print('\nBioBERTContextEncoder test passed!')