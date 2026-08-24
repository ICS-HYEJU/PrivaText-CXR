from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from contextlib import nullcontext

class BioBERTEmbedder(nn.Module):
    """
    BioBERT-based text embedder for radiology reports.

    Input  : list[str]  - report texts (FINDINGS: ~ IMPRESSION: ~)
    Output : Tensor [B, seq_len, output_dim]

    Args:
        model_path : local path or HuggingFace model ID
                     e.g. "/storage/hjchoi/biobert"
                          "dmis-lab/biobert-base-cased-v1.2"
        output_dim : projection output dim; must match UNet context_dim (default 512)
        freeze     : freeze BioBERT weights (True for DP fine-tuning)
        max_length : tokeniser max length in tokens (default 128)
    """

    def __init__(
        self,
        model_path : str,
        output_dim : int  = 512,
        freeze     : bool = True,
        max_length : int  = 128,
    ):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert      = AutoModel.from_pretrained(model_path)
        self.proj      = nn.Linear(768, output_dim)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        n_proj = sum(p.numel() for p in self.proj.parameters())
        print(f"[BioBERTEmbedder] model={model_path}  "
              f"output_dim={output_dim}  freeze={freeze}  "
              f"proj_params={n_proj:,}")

    @property
    def device(self):
        return next(self.bert.parameters()).device

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts : list[str]
        Returns:
            Tensor [B, seq_len, output_dim]
        """
        enc = self.tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
        ).to(self.device)
        ctx = nullcontext() if self.bert.training else torch.no_grad()
        with ctx:
            out = self.bert(**enc)

        return self.proj(out.last_hidden_state)
False