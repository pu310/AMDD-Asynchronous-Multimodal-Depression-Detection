"""MSA audio encoder with GSA attention and temporal pyramid features."""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from .base import BaseNet


def calc_reindexing_tensor(l, L, device):
    x = torch.arange(l, device=device)[:, None, None]
    i = torch.arange(l, device=device)[None, :, None]
    r = torch.arange(-(L - 1), L, device=device)[None, None, :]
    mask = ((i - x) == r) & ((i - x).abs() <= L)
    return mask.float()


class GSA(nn.Module):
    def __init__(self, dim, *, rel_pos_length=None, dim_out=None, heads=8, dim_key=64, norm_queries=False, batch_norm=True):
        super().__init__()
        self.dim_out = dim_out if dim_out is not None else dim
        dim_hidden = dim_key * heads
        self.heads, self.rel_pos_length, self.norm_queries = heads, rel_pos_length, norm_queries
        self.to_qkv = nn.Conv1d(dim, dim_hidden * 3, 1, bias=False)
        self.to_out = nn.Conv1d(dim_hidden, self.dim_out, 1)
        if rel_pos_length is not None:
            num_rel_shifts = 2 * rel_pos_length - 1
            self.norm = nn.BatchNorm1d(dim_key) if batch_norm else None
            self.rel_positions = nn.Parameter(torch.randn(num_rel_shifts, dim_key))
        else:
            self.norm = None

    def forward(self, x):
        b, t, f, h, L, device = *x.shape, self.heads, self.rel_pos_length, x.device
        x = x.transpose(1, 2)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) t -> (b h) c t', h=h), qkv)
        k = k.softmax(dim=-1)
        context = einsum('ndt,net->nde', k, v)
        content_q = q if not self.norm_queries else q.softmax(dim=-2)
        content_out = einsum('nde,ndt->net', context, content_q)
        if self.rel_pos_length is not None:
            It = calc_reindexing_tensor(t, L, device)
            Pt = einsum('tir,rd->tid', It, self.rel_positions)
            St = einsum('ndt,tid->nit', q, Pt)
            rel_pos_out = einsum('nit,net->net', St, v)
            if self.norm is not None:
                rel_pos_out = self.norm(rel_pos_out)
            content_out = content_out + rel_pos_out
        content_out = rearrange(content_out, '(b h) c t -> b (h c) t', h=h)
        return self.to_out(content_out)


class TemporalPyramid(nn.Module):
    """Multi-scale temporal dynamics via dilated convolutions."""

    def __init__(self, in_channels, out_channels, pyramid_channels=64):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, pyramid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(pyramid_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, pyramid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(pyramid_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, pyramid_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(pyramid_channels),
            nn.ReLU()
        )
        self.fusion = nn.Conv1d(pyramid_channels * 3, out_channels, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out_cat = torch.cat([out1, out2, out3], dim=1)
        return F.relu(self.bn_fusion(self.fusion(out_cat)))


class MSAEncoder(BaseNet):
    def __init__(self, hidden_sizes=[256, 256], dropout=0.5, gsa_input=25, gsa_rel_pos_length=10):
        super().__init__()
        self.gsa = GSA(dim=gsa_input, rel_pos_length=gsa_rel_pos_length)
        self.temporal_pyramid = TemporalPyramid(in_channels=gsa_input, out_channels=256)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(96)
        self.classifier_msa = nn.Sequential(
            nn.Linear(256 * 96, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.apply(self.custom_weights_init)

    def custom_weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def feature_extractor(self, x):
        gsa_out = self.gsa(x)
        temporal_features = self.temporal_pyramid(gsa_out)
        pooled_features = self.adaptive_pool(temporal_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return temporal_features, pooled_features

    def classifier(self, x):
        msa_sequence_features, msa_pooled_features = x
        return msa_sequence_features, self.classifier_msa(msa_pooled_features)
