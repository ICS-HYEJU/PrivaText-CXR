from abc import abstractmethod

import math
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from nnFormer import *
from Model.SwinUnet import *
from Model.util_network import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from monai.utils import ensure_tuple_rep


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
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if dims == 3:
            self.sample_kernel = (sample_kernel[0], sample_kernel[1], sample_kernel[2])
        else:
            self.sample_kernel = (sample_kernel[0], sample_kernel[1])
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
        else:
            self.up = torch.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
            self.conv = conv_nd(dims, self.channels, self.channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.up(x)
        x = self.conv(x)

        # if self.dims == 3:
        #     x = F.interpolate(
        #         x, scale_factor=self.sample_kernel, mode="nearest"
        #     )
        # else:
        #     x = F.interpolate(x, scale_factor=self.sample_kernel, mode="nearest")
        # if self.use_conv:
        #     x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if self.dims == 3:
            self.sample_kernel = (1 / sample_kernel[0], 1 / sample_kernel[1], 1 / sample_kernel[2])
        else:
            self.sample_kernel = (1 / sample_kernel[0], 1 / sample_kernel[1])
        # stride = 2 if dims != 3 else (2, 2, 2)
        # stride = 2E
        if use_conv:
            self.op = torch.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
        else:
            assert self.channels == self.out_channels
            self.op = torch.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
            self.conv = conv_nd(dims, self.channels, self.channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # x = F.interpolate(
        #     x, scale_factor=self.sample_kernel, mode="nearest"
        # )
        return self.conv(self.op(x))


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
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
            sample_kernel=None,
            use_swin=False,
            num_heads=4,
            window_size=[4, 4, 4],
            input_resolution=[1, 1, 1],
            drop_path=0.1
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.input_resolution = input_resolution
        self.use_swin = use_swin
        self.dims = dims
        self.window_size = window_size
        self.use_swin = use_swin
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, sample_kernel, dims)
            self.x_upd = Upsample(channels, False, sample_kernel, dims)
        elif down:
            self.h_upd = Downsample(channels, False, sample_kernel, dims)
            self.x_upd = Downsample(channels, False, sample_kernel, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if use_swin:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )

            self.shift_size = tuple(i // 2 for i in window_size)
            self.no_shift = tuple(0 for i in window_size)
            # self.swin_layer = nn.ModuleList([
            #                     SwinTransformerBlock(
            #                         dim=self.out_channels,
            #                         input_resolution=self.input_resolution,
            #                         num_heads=num_heads,
            #                         window_size=window_size,
            #                         shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
            #                         mlp_ratio=4,
            #                         qkv_bias=True,
            #                         qk_scale=None,
            #                         drop=0,
            #                         attn_drop=0,
            #                         drop_path=drop_path,
            #                         norm_layer = nn.LayerNorm)
            #                     for i in range(2)])
            self.swin_layer = nn.ModuleList([SwinTransformerBlock(
                dim=self.out_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0,
                attn_drop=0,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,
                use_checkpoint=None)
                for i in range(2)])
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.Identity())
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            self.swin_layer = nn.ModuleList([nn.Identity()])
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=0),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
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

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x_in, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x_in)
            h = self.h_upd(h)
            x_in = self.x_upd(x_in)
            h = in_conv(h)
        else:
            h_in = self.in_layers(x_in)

        emb_out = self.emb_layers(emb).type(h_in.dtype)
        while len(emb_out.shape) < len(h_in.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h_in = out_norm(h_in) * (1 + scale) + shift

            if self.use_swin:
                if self.dims == 3:
                    b, c, d, h, w = x_in.shape
                    window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
                    h_in = rearrange(h_in, "b c d h w -> b d h w c")
                    dp = int(np.ceil(d / window_size[0])) * window_size[0]
                    hp = int(np.ceil(h / window_size[1])) * window_size[1]
                    wp = int(np.ceil(w / window_size[2])) * window_size[2]
                    attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
                    for blk in self.swin_layer:
                        h_in = blk(h_in, attn_mask)
                    h_in = h_in.view(b, d, h, w, -1)
                    h_in = rearrange(h_in, "b d h w c -> b c d h w")

                elif self.dims == 2: # 2d
                    b, c, h, w = h_in.shape
                    window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
                    h_in = rearrange(h_in, "b c h w -> b h w c")
                    hp = int(np.ceil(h / window_size[0])) * window_size[0]
                    wp = int(np.ceil(w / window_size[1])) * window_size[1]
                    attn_mask = compute_mask([hp, wp], window_size, shift_size, h_in.device)
                    for blk in self.swin_layer:
                        h_in = blk(h_in, attn_mask)
                    h_in = h_in.view(b, h, w, -1)
                    h_in = rearrange(h_in, "b h w c -> b c h w")
            else:
                for blk in self.swin_layer:
                    h_in = blk(h_in)

            # S, H, W = h.size(2), h.size(3), h.size(4)
            # h = h.flatten(2).transpose(1, 2).contiguous()
            # for blk in self.swin_layer:
            #     h = blk(h)
            # h = h.transpose(1, 2).contiguous().view(-1, self.out_channels, S, H, W)

            h_in = out_rest(h_in)
        else:
            h_in = h_in + emb_out

            if self.use_swin:
                if self.dims == 3:
                    b, c, d, h, w = x_in.shape
                    window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
                    h_in = rearrange(h_in, "b c d h w -> b d h w c")
                    dp = int(np.ceil(d / window_size[0])) * window_size[0]
                    hp = int(np.ceil(h / window_size[1])) * window_size[1]
                    wp = int(np.ceil(w / window_size[2])) * window_size[2]
                    attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
                    for blk in self.swin_layer:
                        h_in = blk(h_in, attn_mask)
                    h_in = h_in.view(b, d, h, w, -1)
                    h_in = rearrange(h_in, "b d h w c -> b c d h w")

                elif self.dims == 2:
                    b, c, h, w = h_in.shape
                    window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
                    h_in = rearrange(h_in, "b c h w -> b h w c")
                    hp = int(np.ceil(h / window_size[0])) * window_size[0]
                    wp = int(np.ceil(w / window_size[1])) * window_size[1]
                    attn_mask = compute_mask([hp, wp], window_size, shift_size, h_in.device)
                    for blk in self.swin_layer:
                        h_in = blk(h_in, attn_mask)
                    h_in = h_in.view(b, h, w, -1)
                    h_in = rearrange(h_in, "b h w c -> b c h w")
            else:
                for blk in self.swin_layer:
                    h_in = blk(h_in)

            h_in = self.out_layers(h_in)
        return self.skip_connection(x_in) + h_in


class SwinVITModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=False,
            dims=2,
            sample_kernel=None,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            window_size=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.sample_kernel = sample_kernel
        spatial_dims = dims
        drop_path = [x.item() for x in torch.linspace(0, dropout, len(channel_mult))]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin=use_swin,
                        num_heads=num_heads[level],
                        window_size=window_size[level],
                        input_resolution=ds,
                        drop_path=drop_path[level]
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin=use_swin,
                            num_heads=num_heads[level],
                            window_size=window_size[level],
                            input_resolution=ds,
                            drop_path=drop_path[level],
                            down=True,
                            sample_kernel=self.sample_kernel[level],
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, self.sample_kernel[level], dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                if dims == 3:
                    ds = [ds[0] // self.sample_kernel[level][0], ds[1] // self.sample_kernel[level][1],
                          ds[2] // self.sample_kernel[level][2]]
                else:
                    ds = [ds[0] // self.sample_kernel[level][0], ds[1] // self.sample_kernel[level][1]]
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin=use_swin,
                num_heads=num_heads[level],
                window_size=window_size[level],
                input_resolution=ds,
                drop_path=drop_path[level]
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin=use_swin,
                num_heads=num_heads[level],
                window_size=window_size[level],
                input_resolution=ds,
                drop_path=drop_path[level]
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin=use_swin,
                        num_heads=num_heads[level],
                        window_size=window_size[level],
                        input_resolution=ds,
                        drop_path=drop_path[level]
                    )
                ]
                ch = int(model_channels * mult)

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin=use_swin,
                            num_heads=num_heads[level],
                            window_size=window_size[level],
                            input_resolution=ds,
                            drop_path=drop_path[level],
                            up=True,
                            sample_kernel=self.sample_kernel[level - 1],
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, self.sample_kernel[level - 1], dims=dims, out_channels=out_ch)
                    )
                    if dims == 3:
                        ds = [ds[0] * self.sample_kernel[level - 1][0],
                              ds[1] * self.sample_kernel[level - 1][1],
                              ds[2] * self.sample_kernel[level - 1][2]]
                    else:
                        ds = [ds[0] * self.sample_kernel[level - 1][0],
                              ds[1] * self.sample_kernel[level - 1][1]]
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=2.,
            **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, timesteps, cond=None, null_cond_prob=0., y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # [2,512]

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default="/storage/hjchoi/archive/image_file")
    parser.add_argument("-label_path", default="/storage/hjchoi/archive/Data_Entry_2017.csv")
    parser.add_argument("-task", default="train", choices=["train", "val", "test"])
    parser.add_argument("-image_size", default=256, type=int)
    parser.add_argument("-batch_size", default=2, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-image_show", action="store_true")  # ������ ���� ����
    parser.add_argument("-device", default="cuda:1")
    return parser.parse_args()


def build_model(image_size):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] device:", device)

    num_channels = 128
    channel_mult = (1, 1, 2, 2, 4, 4)
    attention_resolutions = "64,32,16,8"
    num_heads = [4, 4, 4, 8, 16, 16]
    window_size = [[4, 4], [4, 4], [4, 4], [8, 8], [8, 8], [4, 4]]
    num_res_blocks = [2, 2, 1, 1, 1, 1]


    sample_kernel = ([2, 2], [2, 2], [2, 2], [2, 2], [2, 2])

    attention_ds = [int(r) for r in attention_resolutions.split(",")]
    class_cond = False
    use_scale_shift_norm = True
    resblock_updown = False

    model = SwinVITModel(
        image_size=(image_size, image_size),
        in_channels=1,
        model_channels=num_channels,
        out_channels=2,
        sample_kernel=sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=(None if not class_cond else 10),  # ? class_cond=False�� None
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size=window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=False,
    ).to(device)

    model.eval()
    return model, device


if __name__ == "__main__":
    args = parse_args()
    from Data.dataset import *
    # 1) dataset / loader
    dataset = NIH(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 2) Model INIT
    model, device = build_model(args.image_size)

    # 3) Model Forward
    batch = next(iter(loader))
    x, label_str = batch  # x: (B,1,H,W), label_str: list[str]

    print("[DEBUG] x.shape:", x.shape, "x.dtype:", x.dtype)
    print("[DEBUG] label_str[0]:", label_str[0])

    x = x.to(device, non_blocking=True)

    # Diffusion Timestep
    # 0~999
    timesteps = torch.randint(low=0, high=1000, size=(x.shape[0],), device=device, dtype=torch.long)
    print("[DEBUG] timesteps:", timesteps)

    with torch.no_grad():
        try:
            y = model(x, timesteps)  # cond=None, y=None
            print("[DEBUG] output.shape:", y.shape, "output.dtype:", y.dtype)
        except Exception as e:
            print("\n[ERROR] forward failed:", repr(e))
            print("[DEBUG] x:", x.shape, x.dtype, x.device)
            print("[DEBUG] timesteps:", timesteps.shape, timesteps.dtype, timesteps.device)
            raise
