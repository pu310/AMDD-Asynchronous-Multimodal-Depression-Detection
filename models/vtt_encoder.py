"""VTT visual encoder: hierarchical attention blocks with optional temporal Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, max_seq_len=256, dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, t, c = x.shape
        assert t <= self.pos_embedding.num_embeddings, (
            f"Sequence length {t} > max sequence length {self.pos_embedding.num_embeddings}"
        )
        position_ids = torch.arange(t, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(position_ids)
        x = self.transformer_encoder(x)
        return self.layer_norm(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, drop=0.):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(-2, -1).reshape(B, C, L)
        return self.drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, expansion=4,
                 drop=0., drop_path=0., kernel_size=3, act_layer=nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio, drop=drop)
        self.conv = InvertedResidual(
            dim, hidden_dim=dim * expansion, out_dim=dim,
            kernel_size=kernel_size, drop=drop, act_layer=act_layer
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=1, stride=2)
        self.norm = nn.GroupNorm(1, out_dim, eps=1e-6)

    def forward(self, x):
        return self.norm(self.conv(x))


class VTTEncoder(nn.Module):
    """Hierarchical visual sequence encoder with optional temporal Transformer."""

    def __init__(
        self, seq_len=1356, in_chans=136, num_classes=1, dims=[64, 128, 256, 512],
        head_dim=64, expansions=[4, 4, 6, 6], grid_sizes=[1, 1, 1, 1],
        ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], drop_rate=0.,
        drop_path_rate=0., act_layer=nn.SiLU, kernel_sizes=[3, 3, 3, 3],
        use_temporal_transformer=True, tt_depth=2, tt_heads=6, tt_mlp_dim_ratio=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_chans, 16, 3, padding=1, stride=2),
            nn.GroupNorm(1, 16, eps=1e-6),
            act_layer(inplace=True),
            nn.Conv1d(16, dims[0], 3, padding=1, stride=2),
        )

        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for stage in range(len(dims)):
            blocks_stage = nn.ModuleList([
                Block(
                    dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage],
                    expansion=expansions[stage], drop=drop_rate, drop_path=dpr[cur + i],
                    kernel_size=kernel_sizes[stage], act_layer=act_layer
                ).to(device)
                for i in range(depths[stage])
            ])
            self.blocks.append(blocks_stage)
            cur += depths[stage]

        self.ds2 = Downsample(dims[0], dims[1])
        self.ds3 = Downsample(dims[1], dims[2])
        self.ds4 = Downsample(dims[2], dims[3])
        self.use_temporal_transformer = use_temporal_transformer
        if self.use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                dim=dims[-1],
                depth=tt_depth,
                heads=tt_heads,
                mlp_dim=dims[-1] * tt_mlp_dim_ratio,
                max_seq_len=256
            ).to(device)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[-1], num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks[0]:
            x = block(x)
        x = self.ds2(x)

        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)

        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)

        for block in self.blocks[3]:
            x3_block = block(x)

        if self.use_temporal_transformer:
            x_seq = x.permute(0, 2, 1)
            x_seq = self.temporal_transformer(x_seq)
            x = x_seq.permute(0, 2, 1)

        x_adAvg = F.adaptive_avg_pool1d(x3_block, 1).flatten(1)
        x_vtt_cls = self.classifier(x_adAvg)
        return x3_block, x_vtt_cls
