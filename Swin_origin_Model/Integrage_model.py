from abc import abstractmethod
import torch.nn as nn
from Swin_origin_Model.SwinUnet import *
from Model.util_network import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)


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
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
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

        if use_conv:
            self.op = torch.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
        else:
            assert self.channels == self.out_channels
            self.op = torch.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
            self.conv = conv_nd(dims, self.channels, self.channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.conv(self.op(x))


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
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


# ============================================================================
# ENCODER
# ============================================================================
class SwinVITEncoder(nn.Module):
    """Encoder part of SwinVITModel"""

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
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
            use_scale_shift_norm=False,
            resblock_updown=False,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
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
        self.sample_kernel = sample_kernel
        self.dims = dims
        self.resblock_updown = resblock_updown

        drop_path = [x.item() for x in torch.linspace(0, dropout, len(channel_mult))]

        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        self.input_block_chans = [ch]
        ds = list(image_size) if isinstance(image_size, tuple) else [image_size, image_size]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
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
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                self.input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                if ds[0] in attention_resolutions:
                    use_swin = True
                else:
                    use_swin = False

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
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
                self.input_block_chans.append(ch)
                if dims == 3:
                    ds = [ds[0] // self.sample_kernel[level][0], ds[1] // self.sample_kernel[level][1],
                          ds[2] // self.sample_kernel[level][2]]
                else:
                    ds = [ds[0] // self.sample_kernel[level][0], ds[1] // self.sample_kernel[level][1]]
                self._feature_size += ch

        self.output_channels = ch
        self.final_resolution = ds

    def forward(self, x, timesteps, y=None):
        """
        Returns:
            h: bottleneck features
            hs: skip connection features (list)
            emb: time embedding
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        return h, hs, emb


# ============================================================================
# DECODER
# ============================================================================
class SwinVITDecoder(nn.Module):
    """Decoder part of SwinVITModel"""

    def __init__(
            self,
            image_size,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=False,
            dims=2,
            sample_kernel=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            window_size=4,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder=None,  # Encoder ������ ������ ������ ������
    ):
        super().__init__()

        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dims = dims
        self.resblock_updown = resblock_updown

        drop_path = [x.item() for x in torch.linspace(0, dropout, len(channel_mult))]
        time_embed_dim = model_channels * 4

        # Encoder���� ���� ��������
        if encoder is not None:
            input_block_chans = encoder.input_block_chans.copy()
            ch = encoder.output_channels
            ds = encoder.final_resolution.copy()
        else:
            # Encoder ���� ���������� ���������� ����
            ch = int(channel_mult[-1] * model_channels)
            ds = [image_size[0] // (2 ** (len(channel_mult) - 1)),
                  image_size[1] // (2 ** (len(channel_mult) - 1))]
            input_block_chans = None

        input_ch = int(channel_mult[0] * model_channels)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop() if input_block_chans else ch
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
                            sample_kernel=sample_kernel[level - 1],
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, sample_kernel[level - 1], dims=dims, out_channels=out_ch)
                    )
                    if dims == 3:
                        ds = [ds[0] * sample_kernel[level - 1][0],
                              ds[1] * sample_kernel[level - 1][1],
                              ds[2] * sample_kernel[level - 1][2]]
                    else:
                        ds = [ds[0] * sample_kernel[level - 1][0],
                              ds[1] * sample_kernel[level - 1][1]]

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, h, hs, emb, x_dtype):
        """
        Args:
            h: bottleneck features from middle block
            hs: skip connection features from encoder (list)
            emb: time embedding from encoder
            x_dtype: original input dtype
        """
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x_dtype)
        return self.out(h)


class UNetMiddleBlock(nn.Module):
    """
    UNetModel�� middle block���� �������� ���� Wrapper
    ���� UNetModel�� ���� forward�� middle block���� ���������� ����
    """

    def __init__(
            self,
            # UNetModel�� ���� ����������
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            # Middle block ���� ����
            adapt_input_channels=False,  # ���� ���� ���� �������� ������ ����
            adapt_output_channels=False,  # ���� ���� ���� �������� ������ ����
    ):
        super().__init__()

        self.unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
        )

        self.adapt_input_channels = adapt_input_channels
        self.adapt_output_channels = adapt_output_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ���� ���� ������ (������ ����)
        if adapt_input_channels:
            self.input_adapter = None  # forward���� �������� ����

        # ���� ���� ������ (������ ����)
        if adapt_output_channels:
            self.output_adapter = None  # forward���� �������� ����

    def forward(self, h, emb, context=None):
        """
        Middle block���� ����
        Args:
            h: [B, C, H, W] - Encoder�� bottleneck features
            emb: [B, emb_dim] - Time embedding from encoder
            context: Optional cross-attention context
        Returns:
            h: [B, C, H, W] - Processed features
        """
        B, C, H, W = h.shape

        # ���� ���� ������ (������ ����)
        if self.adapt_input_channels and C != self.in_channels:
            if self.input_adapter is None:
                self.input_adapter = nn.Conv2d(C, self.in_channels, 1).to(h.device)
            h = self.input_adapter(h)

        # Timestep ���� (UNet�� timestep�� ������ ��)
        # emb���� ������ timestep�� ����������, ���� timestep ����
        device = h.device
        timesteps = torch.zeros(B, dtype=torch.long, device=device)  # ���� timestep (0���� ����)

        # UNet forward
        h_out = self.unet(h, timesteps=timesteps, context=context)

        # ���� ���� ������ (������ ����)
        if self.adapt_output_channels and h_out.shape[1] != C:
            if self.output_adapter is None:
                self.output_adapter = nn.Conv2d(h_out.shape[1], C, 1).to(h.device)
            h_out = self.output_adapter(h_out)

        return h_out

# ============================================================================
# INTEGRATED MODEL
# ============================================================================
class IntegratedModel(nn.Module):
    """���� ����: Encoder + Custom Middle Block + Decoder"""

    def __init__(self, encoder_config, middle_block, decoder_config):
        super().__init__()

        # Encoder ������
        self.encoder = SwinVITEncoder(**encoder_config)

        # Middle block (������ ���� ����)
        self.middle_block = middle_block

        # Decoder ������ (encoder ���� ����)
        decoder_config['encoder'] = self.encoder
        self.decoder = SwinVITDecoder(**decoder_config)

    def forward(self, x, timesteps, y=None):
        # Encoder
        h, hs, emb = self.encoder(x, timesteps, y)

        # Middle Block (������ ����)
        h = self.middle_block(h, emb)

        # Decoder
        output = self.decoder(h, hs, emb, x.dtype)

        return output


# ============================================================================
# ARGUMENT PARSING & MODEL BUILDING
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default="/storage/hjchoi/archive/image_file")
    parser.add_argument("-label_path", default="/storage/hjchoi/archive/Data_Entry_2017.csv")
    parser.add_argument("-task", default="train", choices=["train", "val", "test"])
    parser.add_argument("-image_size", default=256, type=int)
    parser.add_argument("-batch_size", default=2, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-image_show", action="store_true")
    parser.add_argument("-device", default="cuda:1")
    parser.add_argument("-test_mode", default="encoder_only", choices=["integrated", "encoder_only", "full_pipeline"])
    return parser.parse_args()


def build_encoder_only(image_size, device):
    """Encoder�� ������"""
    print("[DEBUG] Building Encoder Only...")

    num_channels = 128
    channel_mult = (1, 1, 2, 2, 4)
    attention_resolutions = [64, 32, 16]
    num_heads = [4, 4, 4, 8, 16]
    window_size = [[4, 4], [4, 4], [4, 4], [8, 8], [8, 8]]
    num_res_blocks = [2, 2, 1, 1, 1]
    sample_kernel = ([2, 2], [2, 2], [2, 2], [2, 2])

    encoder = SwinVITEncoder(
        image_size=(image_size, image_size),
        in_channels=1,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_resolutions),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size=window_size,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=False,
        dims=2,
        sample_kernel=sample_kernel,
    ).to(device)

    encoder.eval()
    return encoder


def build_integrated_model(image_size, device):
    """���� ���� ����"""
    print("[DEBUG] Building Integrated Swin_origin_Model (Encoder + Middle + Decoder)...")

    # ���� ����
    num_channels = 128
    channel_mult = (1, 1, 2, 2, 4)
    attention_resolutions = [64, 32, 16]
    num_heads = [4, 4, 4, 8, 16]
    window_size = [[4, 4], [4, 4], [4, 4], [8, 8], [8, 8]]
    num_res_blocks = [2, 2, 1, 1, 1]
    sample_kernel = ([2, 2], [2, 2], [2, 2], [2, 2])

    # Encoder ����
    encoder_config = {
        'image_size': (image_size, image_size),
        'in_channels': 1,
        'model_channels': num_channels,
        'num_res_blocks': num_res_blocks,
        'attention_resolutions': tuple(attention_resolutions),
        'dropout': 0,
        'channel_mult': channel_mult,
        'num_classes': None,
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': num_heads,
        'window_size': window_size,
        'num_head_channels': 64,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'dims': 2,
        'sample_kernel': sample_kernel,
    }

    middle_block = TimestepEmbedSequential(

    )

    # Decoder ����
    decoder_config = {
        'image_size': (image_size, image_size),
        'model_channels': num_channels,
        'out_channels': 2,
        'num_res_blocks': num_res_blocks,
        'attention_resolutions': tuple(attention_resolutions),
        'dropout': 0,
        'channel_mult': channel_mult,
        'use_checkpoint': False,
        'use_fp16': False,
        'num_heads': num_heads,
        'window_size': window_size,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'dims': 2,
        'sample_kernel': sample_kernel,
    }

    model = IntegratedModel(encoder_config, middle_block, decoder_config).to(device)
    model.eval()

    return model


# ============================================================================
# MAIN - DEBUGGING
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    from Data.dataset import *

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] device: {device}")
    print(f"[DEBUG] test_mode: {args.test_mode}")

    # 1) Dataset / Loader
    print("\n[DEBUG] Loading Dataset...")
    dataset = NIH(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 2) Get one batch
    print("[DEBUG] Getting one batch...")
    batch = next(iter(loader))
    x, label_str = batch  # x: (B,1,H,W), label_str: list[str]

    print(f"[DEBUG] x.shape: {x.shape}, x.dtype: {x.dtype}")
    print(f"[DEBUG] label_str[0]: {label_str[0]}")

    x = x.to(device, non_blocking=True)

    # Swin_origin_Diffusion Timestep (0~999)
    timesteps = torch.randint(low=0, high=1000, size=(x.shape[0],), device=device, dtype=torch.long)
    print(f"[DEBUG] timesteps: {timesteps}")

    # 3) Swin_origin_Model Testing
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] Testing Mode: {args.test_mode}")
    print(f"{'=' * 60}")

    with torch.no_grad():
        try:
            if args.test_mode == "encoder_only":
                # Encoder�� ������
                encoder = build_encoder_only(args.image_size, device)
                print(f"[DEBUG] Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

                h, hs, emb = encoder(x, timesteps)

                print(f"\n[DEBUG] Encoder Forward Success!")
                print(f"[DEBUG] Bottleneck h.shape: {h.shape}, h.dtype: {h.dtype}")
                print(f"[DEBUG] Number of skip connections: {len(hs)}")
                print(f"[DEBUG] Skip connection shapes:")
                for i, skip in enumerate(hs):
                    print(f"  - hs[{i}]: {skip.shape}")
                print(f"[DEBUG] Time embedding emb.shape: {emb.shape}")

            elif args.test_mode == "integrated":
                # ���� ���� ������ (Encoder + Middle + Decoder)
                model = build_integrated_model(args.image_size, device)
                print(f"[DEBUG] Integrated Swin_origin_Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(f"[DEBUG] - Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
                print(f"[DEBUG] - Middle Block parameters: {sum(p.numel() for p in model.middle_block.parameters()):,}")
                print(f"[DEBUG] - Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")

                output = model(x, timesteps)

                print(f"\n[DEBUG] Integrated Swin_origin_Model Forward Success!")
                print(f"[DEBUG] output.shape: {output.shape}, output.dtype: {output.dtype}")
                print(f"[DEBUG] output range: [{output.min():.4f}, {output.max():.4f}]")

            elif args.test_mode == "full_pipeline":
                # ���� ���������� ������ (������)
                model = build_integrated_model(args.image_size, device)

                print("\n[Step 1] Encoder Forward...")
                h, hs, emb = model.encoder(x, timesteps)
                print(f"   Encoder output h.shape: {h.shape}")
                print(f"   Skip connections: {len(hs)} layers")

                print("\n[Step 2] Middle Block Forward...")
                h_middle = model.middle_block(h, emb)
                print(f"  Middle block output h.shape: {h_middle.shape}")

                print("\n[Step 3] Decoder Forward...")
                output = model.decoder(h_middle, hs, emb, x.dtype)
                print(f"  ? Decoder output shape: {output.shape}")

                print(f"\n[DEBUG] ? Full Pipeline Success!")
                print(f"[DEBUG] Final output shape: {output.shape}, dtype: {output.dtype}")
                print(f"[DEBUG] output range: [{output.min():.4f}, {output.max():.4f}]")

        except Exception as e:
            print(f"\n[ERROR] Forward failed: {repr(e)}")
            print(f"[DEBUG] x: {x.shape}, {x.dtype}, {x.device}")
            print(f"[DEBUG] timesteps: {timesteps.shape}, {timesteps.dtype}, {timesteps.device}")
            import traceback

            traceback.print_exc()
            raise

    print(f"\n{'=' * 60}")
    print("[DEBUG] All tests completed successfully!")
    print(f"{'=' * 60}")