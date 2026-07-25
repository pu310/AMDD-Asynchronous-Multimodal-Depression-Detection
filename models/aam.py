"""Asynchronous Association Module (AAM) for audio-visual cross-attention fusion."""

import torch
import torch.nn as nn


class AAM(nn.Module):
    """Bidirectional cross-attention with temporal shift for audio-visual fusion."""

    def __init__(self, d=256, f_d=512, l=8, temporal_kernel_size=7):
        super().__init__()

        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        self.qa_transform = nn.Linear(d, d)
        self.qv_transform = nn.Linear(d, d)
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)

        self.qf_transform = nn.Linear(f_d, d)
        self.kf_transform = nn.Linear(f_d, d)
        self.vf_transform = nn.Linear(f_d, d)

        self.temporal_shift_v = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=temporal_kernel_size, padding='same'
        )
        self.temporal_shift_a = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=temporal_kernel_size, padding='same'
        )

        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

    def forward(self, v, a):
        v_encoded = self.v_encoder(v)
        a_encoded = self.a_encoder(a)

        q_a = self.qa_transform(a_encoded)
        k_from_v = self.kv_transform(v_encoded)
        v_from_v = self.vv_transform(v_encoded)
        k_from_v_shifted = self.temporal_shift_v(k_from_v.permute(0, 2, 1)).permute(0, 2, 1)
        v_from_v_shifted = self.temporal_shift_v(v_from_v.permute(0, 2, 1)).permute(0, 2, 1)
        fav = self.cross_av(q_a, k_from_v_shifted, v_from_v_shifted)[0]

        q_v = self.qv_transform(v_encoded)
        k_from_a = self.ka_transform(a_encoded)
        v_from_a = self.va_transform(a_encoded)
        k_from_a_shifted = self.temporal_shift_a(k_from_a.permute(0, 2, 1)).permute(0, 2, 1)
        v_from_a_shifted = self.temporal_shift_a(v_from_a.permute(0, 2, 1)).permute(0, 2, 1)
        fva = self.cross_va(q_v, k_from_a_shifted, v_from_a_shifted)[0]

        fused_a_v = torch.cat([v_encoded, a_encoded], dim=2)
        q_f = self.qf_transform(fused_a_v)
        k_f = self.kf_transform(fused_a_v)
        v_f = self.vf_transform(fused_a_v)
        f_a_v = self.cross_va(q_f, k_f, v_f)[0]

        return torch.cat([fav, fva, f_a_v], dim=2)
