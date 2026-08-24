'''
Shared Block: ResBlock, DownSample, UpSample
'''

import torch
import torch.nn as nn
from Model.timestep_block import *
from Model.util_network import *


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Supports both timestep-conditioned (UNet) and non-conditioned (Encoder) usage.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Only create emb_layers if emb_channels > 0
        if self.emb_channels > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        else:
            self.emb_layers = None

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: input tensor [B, C, ...]
        :param emb: timestep embedding [B, emb_channels] or None for Encoder
        :return: output tensor [B, out_channels, ...]
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # Process timestep embedding
        if self.emb_layers is not None and emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        else:
            # No timestep embedding (Encoder case)
            emb_out = 0

        # Apply scale-shift normalization or simple addition
        if self.use_scale_shift_norm and self.emb_layers is not None and emb is not None:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if isinstance(emb_out, int):  # emb_out == 0
                h = h  # Don't add anything
            else:
                h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class Upsample(nn.Module):
    def __init__(
            self,
            channels,
            use_conv,
            dims=2,
            out_channels=None,
            sample_kernel=None,
            padding=1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if sample_kernel is None:
            self.sample_kernel = 2
        elif isinstance(sample_kernel, (list, tuple)):
            self.sample_kernel = tuple(sample_kernel)
        else:
            self.sample_kernel = sample_kernel

        if use_conv:
            self.conv = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels

        x = self._upsample(x)

        if self.use_conv:
            x = self.conv(x)

        return x

    def _upsample(self, x):
        if self.dims == 3:
            if isinstance(self.sample_kernel, (tuple, list)):
                d_scale, h_scale, w_scale = self.sample_kernel
                new_kernel = (
                    x.shape[2] * d_scale,
                    x.shape[3] * h_scale,
                    x.shape[4] * w_scale
                )
                return F.interpolate(x, size=new_kernel, mode="nearest")
            else:
                return F.interpolate(
                    x,
                    (x.shape[2], x.shape[3] * self.sample_kernel, x.shape[4] * self.sample_kernel),
                    mode="nearest"
                )
        elif self.dims == 2:
            if isinstance(self.sample_kernel, (tuple, list)):
                h_scale, w_scale = self.sample_kernel
                new_h = x.shape[2] * h_scale
                new_w = x.shape[3] * w_scale
                return F.interpolate(x, size=(new_h, new_w), mode="nearest")
            else:
                return F.interpolate(x, scale_factor=self.sample_kernel, mode="nearest")
        else:
            return F.interpolate(x, scale_factor=self.sample_kernel, mode="nearest")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Unified implementation supporting both MT-DDPM and DP-LDM styles.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param out_channels: if specified, the number of out channels.
    :param sample_kernel: downsampling scale factor. Can be:
                         - None: defaults to 2 (DP-LDM style)
                         - int: same scale for all dimensions
                         - tuple/list: per-dimension scale factors (MT-DDPM style)
    :param padding: padding for convolution (default: 1).
    """

    def __init__(
            self,
            channels,
            use_conv,
            dims=2,
            out_channels=None,
            sample_kernel=None,
            padding=1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if sample_kernel is None:
            self.sample_kernel = 2
        elif isinstance(sample_kernel, (list, tuple)):
            if dims == 3:
                assert len(sample_kernel) == 3, "3D requires 3 scale factors"
                self.sample_kernel = tuple(sample_kernel)
            elif dims == 2:
                assert len(sample_kernel) == 2, "2D requires 2 scale factors"
                self.sample_kernel = tuple(sample_kernel)
            else:  # dims == 1
                assert len(sample_kernel) == 1, "1D requires 1 scale factor"
                self.sample_kernel = sample_kernel[0]
        else:
            # int: same scale for all dimensions
            self.sample_kernel = sample_kernel

        if use_conv:
            stride = self.sample_kernel
            self.op = conv_nd(dims,self.channels, self.out_channels,kernel_size=3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels, \
                "out_channels must equal channels when use_conv=False"

            stride = self.sample_kernel
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels, \
            f"Expected {self.channels} channels, got {x.shape[1]}"
        return self.op(x)