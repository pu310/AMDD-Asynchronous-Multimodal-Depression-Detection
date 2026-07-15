# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNet
from .msa_encoder import MSAEncoder
from .vtt_encoder import VTTEncoder
from .async_cross_fusion import AsyncCrossFusion

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AMMD(BaseNet):
    def __init__(self, d=256, l=6, target_seq_len=64): # 【新增】添加目标序列长度作为参数
        super().__init__()
        
        self.target_seq_len = target_seq_len # 保存目标序列长度

        # 音频特征提取器
        self.msa_encoder = MSAEncoder(gsa_input=25, gsa_rel_pos_length=10).to(device)

        # 视频特征提取器
        self.vtt_encoder = VTTEncoder(num_classes=1, in_chans=136 * 2, dims=[48, 96, 240, 384], head_dim=48, 
                             expansions=[8, 8, 4, 4], grid_sizes=[8, 7, 7, 1], 
                             ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3]).to(device)
        
        # 异步融合模块
        self.async_cross_fusion = AsyncCrossFusion(d=d, f_d=2*d, l=l).to(device)

        # 下采样层
        self.audio_downsample = nn.Sequential(
            nn.Conv1d(256, d, kernel_size=1),
            nn.BatchNorm1d(d),
        )
        self.video_downsample = nn.Sequential(
            nn.Conv1d(384, d, kernel_size=1),
            nn.BatchNorm1d(d),
        )

        # ------------------- 【代码修复 START】 -------------------
        # 新增自适应池化层来统一序列长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_seq_len)
        # -------------------- 【代码修复 END】 --------------------

        # 最终融合序列的编码器
        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=3*d, nhead=6, dim_feedforward=6*d, batch_first=True
            ),
            num_layers=l,
        )

        # 分类层
        self.z_dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(3*d, 1)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def feature_extractor(self, x):
        xa_raw = x[:, :, 136:]
        xv_raw = x[:, :, :136]

        # 视频时间差分增强
        xv_diff = xv_raw[:, 1:, :] - xv_raw[:, :-1, :]
        xv_diff = F.pad(xv_diff, (0, 0, 1, 0), "constant", 0)
        xv_enhanced = torch.cat([xv_raw, xv_diff], dim=2)

        # 音频处理
        xa, msa_pooled_features = self.msa_encoder.feature_extractor(xa_raw)
        msa_cls = self.msa_encoder.classifier_msa(msa_pooled_features)

        # 视频处理
        xv_enhanced = xv_enhanced.permute(0, 2, 1)
        xv, vtt_cls = self.vtt_encoder(xv_enhanced)

        # 下采样
        xa = self.audio_downsample(xa).transpose(1, 2) # Shape: [B, T_a, C]
        xv = self.video_downsample(xv).transpose(1, 2) # Shape: [B, T_v, C]

        # ------------------- 【代码修复 START】 -------------------
        # 统一序列长度
        # AdaptiveAvgPool1d 需要 (B, C, T) 的输入，所以先换位
        xa = self.adaptive_pool(xa.permute(0, 2, 1)).permute(0, 2, 1) # Shape: [B, target_len, C]
        xv = self.adaptive_pool(xv.permute(0, 2, 1)).permute(0, 2, 1) # Shape: [B, target_len, C]
        # -------------------- 【代码修复 END】 --------------------

        # 异步融合 (现在xa和xv的序列长度保证一致了)
        xav_cross_feats = self.async_cross_fusion(xv, xa)

        # 最终编码和池化
        xav_fused = self.av_encoder(xav_cross_feats)
        z = torch.mean(xav_fused, dim=1)
        return self.z_dropout(z), msa_cls, vtt_cls

    def classifier(self, x):
        z, msa_cls, vtt_cls = x
        return self.fc(z), msa_cls, vtt_cls
