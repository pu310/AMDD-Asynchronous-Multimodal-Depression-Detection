# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
Source Paper: https://arxiv.org/abs/2401.14185
This version is modified to include Asynchronous Cross-Attention.
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsyncCrossFusion(nn.Module):
    def __init__(self, d=256, f_d=512, l=6, temporal_kernel_size=3):
        """
        Initializes the AsyncCrossFusion class.

        Args:
            d (int)                 :   Dimensionality of the model.
            f_d (int)               :   Fused Dimension from the model (for the self-attention part).
            l (int)                 :   Number of layers in the Transformer encoder.
            temporal_kernel_size(int):  Kernel size for the 1D convolution to model asynchrony.
        """
        super().__init__()

        # Encoders for visual features
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Encoders for audio features
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Projection layers for cross-attention
        ## Audio Query
        self.qa_transform = nn.Linear(d, d)
        ## Video Query
        self.qv_transform = nn.Linear(d, d)
        
        ## Key/Value from Audio
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        ## Key/Value from Video
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)

        # Fused: Audio + Video
        self.qf_transform = nn.Linear(f_d, d)
        self.kf_transform = nn.Linear(f_d, d)
        self.vf_transform = nn.Linear(f_d, d)
        
        # ------------------- 【创新点 START】 -------------------
        # Temporal Convolutional Shift layers to model asynchrony.
        # We apply this to the Key and Value sequences before attention.
        # padding='same' ensures the sequence length does not change.
        
        # Convolutions for when Video provides Key/Value
        self.temporal_shift_v = nn.Conv1d(
            in_channels=d, out_channels=d, 
            kernel_size=temporal_kernel_size, padding='same'
        )
        
        # Convolutions for when Audio provides Key/Value
        self.temporal_shift_a = nn.Conv1d(
            in_channels=d, out_channels=d, 
            kernel_size=temporal_kernel_size, padding='same'
        )
        # -------------------- 【创新点 END】 --------------------


        # Cross-attention layers: audio x video
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

        # Cross-attention layers: video x audio
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

    def forward(self, v, a):
        """
        Forward pass for the AsyncCrossFusion model.

        Args:
            v (torch.Tensor): Input tensor representing visual features, shape [batch_size, seq_length, d].
            a (torch.Tensor): Input tensor representing audio features, shape [batch_size, seq_length, d].

        Returns:
            torch.Tensor: Output tensor after mutual cross-attention and fusion.
        """
        # 1. Initial unimodal encoding
        v_encoded = self.v_encoder(v)
        a_encoded = self.a_encoder(a)

        # 2. Asynchronous Cross-Modal Attention
        
        # MT-1: Audio (q) queries Video (k, v)
        q_a = self.qa_transform(a_encoded)
        k_from_v = self.kv_transform(v_encoded)
        v_from_v = self.vv_transform(v_encoded)

        # ------------------- 【创新点 START】 -------------------
        # Apply temporal shift to video's Key and Value before audio queries them.
        # Conv1d expects (Batch, Channels, Length), so we permute.
        k_from_v_shifted = self.temporal_shift_v(k_from_v.permute(0, 2, 1)).permute(0, 2, 1)
        v_from_v_shifted = self.temporal_shift_v(v_from_v.permute(0, 2, 1)).permute(0, 2, 1)
        # -------------------- 【创新点 END】 --------------------
        
        # Cross Attention with temporally shifted K and V
        fav = self.cross_av(q_a, k_from_v_shifted, v_from_v_shifted)[0]

        # MT-2: Video (q) queries Audio (k, v)
        q_v = self.qv_transform(v_encoded)
        k_from_a = self.ka_transform(a_encoded)
        v_from_a = self.va_transform(a_encoded)

        # ------------------- 【创新点 START】 -------------------
        # Apply temporal shift to audio's Key and Value before video queries them.
        k_from_a_shifted = self.temporal_shift_a(k_from_a.permute(0, 2, 1)).permute(0, 2, 1)
        v_from_a_shifted = self.temporal_shift_a(v_from_a.permute(0, 2, 1)).permute(0, 2, 1)
        # -------------------- 【创新点 END】 --------------------

        # Cross Attention with temporally shifted K and V
        fva = self.cross_va(q_v, k_from_a_shifted, v_from_a_shifted)[0]

        # 3. Self-Attention on Fused Features
        # The original paper's T-3 is essentially self-attention on concatenated features.
        fused_a_v = torch.cat([v_encoded, a_encoded], dim=2)
        q_f = self.qf_transform(fused_a_v)
        k_f = self.kf_transform(fused_a_v)
        v_f = self.vf_transform(fused_a_v)
        # Using cross_va block for self-attention is fine, as it's just a MultiheadAttention layer.
        f_a_v = self.cross_va(q_f, k_f, v_f)[0]

        # 4. Final Concatenation
        fused_features = torch.cat([fav, fva, f_a_v], dim=2)
        
        return fused_features
