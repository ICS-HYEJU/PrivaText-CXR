from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Model.attention_module import *

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input. Also supports optional context for
    spatial transformers (cross-attention).

    Compatible with both MT-DDPM and DP-LDM architectures.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                # If using 'class SpatialTransformer', passing context. (for CrossAttention)
                x = layer(x, context)
            else:
                x = layer(x)
        return x