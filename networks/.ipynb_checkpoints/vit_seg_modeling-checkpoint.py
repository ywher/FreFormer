# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .FreqFusion import FreqFusion

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
class DecoderBlock_fusion(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        # print(f"Input to DecoderBlock - in:{in_channels}, out:{out_channels}, skip:{skip_channels}")
        
        
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        else:
            x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class DecoderCupFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):
                skip_channels[3-i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.freq_fusions = nn.ModuleList()
        for i in range(len(in_channels)):
            self.freq_fusions.append(
                FreqFusion(
                    hr_channels=out_channels[i],
                    lr_channels=in_channels[i],
                    scale_factor=1,
                    lowpass_kernel=5,
                    highpass_kernel=3,
                    up_group=1,
                    compressed_channels=64,  # 动态调整压缩通道
                    feature_resample=True
                )
            )
        
        blocks = [
            DecoderBlock_fusion(in_ch, out_ch, out_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        blocks[-1] = DecoderBlock(in_channels[-1], out_channels[-1], 0)
        self.blocks = nn.ModuleList(blocks)

        # 通道对齐卷积
        self.skip_convs = nn.ModuleList()
        for i in range(len(skip_channels)):
            if skip_channels[i] > 0:
                self.skip_convs.append(
                    nn.Conv2d(skip_channels[i], out_channels[i], kernel_size=1)
                )
            else:
                self.skip_convs.append(None)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        # print(f"\nInitial decoder input shape: {x.shape}")
        # 存储各阶段输出用于可视化
        decoder_outputs = []
        
        for i, (decoder_block, freq_fusion) in enumerate(zip(self.blocks, self.freq_fusions)):
            # print(f"\n--- Decoder Stage {i+1} ---")
            # print("features:",{features is not None}, self.config.n_skip)
            if features is not None and i < self.config.n_skip:
                
                skip = features[i]
                # print(f"Original skip shape: {skip.shape if skip is not None else 'None'}")
                # 通道对齐
                if self.skip_convs[i] is not None:
                    skip = self.skip_convs[i](skip)
                    # print(f"After skip conv shape: {skip.shape}")
                # 频率感知特征融合
                # print(f"Input to FreqFusion - hr_feat: {skip.shape}, lr_feat: {x.shape}")
                _, hr_feat, lr_feat = freq_fusion(hr_feat=skip, lr_feat=x)
                
                # print(f"Output to FreqFusion - hr_feat: {hr_feat.shape}, lr_feat: {lr_feat.shape}")
                # 常规解码块处理
                x = decoder_block(lr_feat, skip=hr_feat)
                # print("x.shape:", {x.shape})
            else:
                # 无跳跃连接时直接上采样
                x = decoder_block(x, skip=None)
                # print("non-skip x.shape:", {x.shape})
            decoder_outputs.append(x)
        
        return x
        
class FusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(attn_out + x_flat)
        mlp_out = self.mlp(x_flat)
        x_flat = self.norm(mlp_out + x_flat)
        return x_flat.permute(0, 2, 1).view(B, C, H, W)
        
class CrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用更小的中间维度
        reduction_ratio = 1  # 可以调整这个值
        self.inner_dim = channels // reduction_ratio
        
        self.query = nn.Sequential(
            nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inner_dim),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inner_dim),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inner_dim),
            nn.ReLU(inplace=True)
        )

        # 输出投影（恢复原始通道数）
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, context):
        B, C, H, W = x.shape
        
        # 生成Q,K,V - 使用更小的inner_dim
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # [B, HW, D]
        k = self.key(context).view(B, -1, H*W)  # [B, D, HW]
        v = self.value(context).view(B, -1, H*W)  # [B, D, HW]
        
        # 缩放点积注意力
        attn = torch.bmm(q, k) * (self.inner_dim ** -0.5)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, D, HW]
        out = out.view(B, self.inner_dim, H, W)
        out = self.out_proj(out)  # 投影回原始通道数
        
        return x + self.gamma * out    
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # [batch_size, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [batch_size, C, 1, 1]
        out = avg_out + max_out
        return self.sigmoid(out)  # [batch_size, C, 1, 1]
        
class VisionTransformer_mixRf(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super().__init__()
        self.config = config
        # 双模态特征提取主干
        self.transformer = Transformer(config, img_size, vis)  # 主分支
        self.transformer_rf = Transformer(config, img_size, vis)   # RF分支
        self.rf_stn = TPSTransformer()
        # self.LinearCrossAttention = LinearCrossAttention()
        # 区域对齐模块（多尺度）
        # self.region_align = nn.ModuleList([
        #     RegionAlignModule(config.hidden_size)
        #     for _ in range(3)  # 在3个不同尺度对齐
        # ])
        
        # # 置信感知融合
        # self.confidence_fusion = ConfidenceAwareFusion(config.hidden_size)
        
        # 解码器系统
        self.decoder = DecoderCupFusion(config)
        
        # 分割头
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        
        # RoI抖动参数（训练时增强）
        # self.roi_jitter_std = 0.1  # 抖动标准差

    def forward(self, x, rf=None):

        if x.size()[1] == 1:
            if rf is not None:
                rf = self.rf_stn(rf)
                x = torch.cat([x+rf, x+rf, x+rf], dim=1)  # 形状变为 (B, 3, H, W)
            else:
                x = x.repeat(1, 3, 1, 1)
        x_img, _, features_img = self.transformer(x)
        x = self.decoder(x_img, features_img)
        logits = self.segmentation_head(x)
        return logits
        
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                        
        with torch.no_grad():

            res_weight = weights
            self.transformer_rf.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer_rf.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer_rf.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_rf.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer_rf.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer_rf.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer_rf.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer_rf.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer_rf.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer_rf.embeddings.hybrid:
                self.transformer_rf.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer_rf.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer_rf.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer_rf.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                        
# class TPSTransformer(nn.Module):
#     def __init__(self, in_channels=1, num_control_points=5):
#         super().__init__()
#         self.num_control_points = num_control_points
        
#         # 特征提取网络
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4))
#         )
        
#         # 预测控制点位移
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Linear(256, 2 * num_control_points)
#         )
        
#         # 初始化控制点（不再注册为buffer）
#         self.control_points = nn.Parameter(self._init_control_points(), requires_grad=False)
    
#     def _init_control_points(self):
#         """初始化均匀分布的控制点网格"""
#         ctrl_pts = torch.zeros(self.num_control_points, 2)
#         for i in range(self.num_control_points):
#             x = (i % int(math.sqrt(self.num_control_points))) / (math.sqrt(self.num_control_points)-1) * 2 - 1
#             y = (i // int(math.sqrt(self.num_control_points))) / (math.sqrt(self.num_control_points)-1) * 2 - 1
#             ctrl_pts[i, 0] = x
#             ctrl_pts[i, 1] = y
#         return ctrl_pts 
        
#     def _apply_tps_transform(self, grid, W, A):
#         """完全修正的TPS变换应用方法"""
#         batch_size, H, W_dim, _ = grid.shape
#         device = grid.device
#         num_points = H * W_dim
    
#         # 1. 准备控制点和网格点
#         points = grid.reshape(batch_size, -1, 2)  # [B, H*W, 2]
#         control_points = self.control_points.to(device)  # [num_ctrl, 2]
    
#         # 2. 计算径向基函数U [B, H*W, num_ctrl]
#         diffs = points.unsqueeze(2) - control_points.unsqueeze(0).unsqueeze(0)  # [B, H*W, num_ctrl, 2]
#         r2 = (diffs ** 2).sum(dim=-1)  # [B, H*W, num_ctrl]
#         U = torch.where(r2 > 0, r2 * torch.log(r2), torch.zeros_like(r2))
    
#         # 3. 计算弯曲部分 [B, H*W, 2]
#         warp = torch.bmm(U, W)  # 使用批量矩阵乘法
    
#         # 4. 计算仿射部分 [B, H*W, 2]
#         affine = A[:, 0:1] + (A[:, 1:2] * points[..., 0:1]) + (A[:, 2:3] * points[..., 1:2])
    
#         # 5. 合并结果
#         transformed = affine + warp
    
#         # 6. 恢复原始形状 [B, H, W, 2]
#         return transformed.reshape(batch_size, H, W_dim, 2)
        
#     def _compute_tps_weights(self, source, target):
#         """计算TPS变换参数（保留梯度版本）"""
#         device = source.device
#         n = source.size(0)
        
#         # 计算距离矩阵（保持梯度）
#         diff = source.unsqueeze(1) - source.unsqueeze(0)  # [n, n, 2]
#         r2 = (diff ** 2).sum(dim=-1)  # [n, n]
        
#         # 避免log(0)的情况
#         safe_r2 = torch.where(r2 > 1e-10, r2, torch.ones_like(r2))
#         K = r2 * torch.log(safe_r2)
        
#         # 构造线性方程组
#         P = torch.cat([torch.ones(n, 1, device=device), source], dim=1)  # [n, 3]
#         L_upper = torch.cat([K, P], dim=1)  # [n, n+3]
#         L_lower = torch.cat([P.T, torch.zeros(3, 3, device=device)], dim=1)  # [3, n+3]
#         L = torch.cat([L_upper, L_lower], dim=0)  # [n+3, n+3]
        
#         # 构造右侧矩阵
#         Y = torch.cat([target, torch.zeros(3, 2, device=device)], dim=0)  # [n+3, 2]
        
#         # 添加小噪声保证矩阵可逆
#         L = L + torch.eye(L.size(0), device=device) * 1e-6
        
#         # 使用可微的矩阵求解
#         try:
#             W = torch.linalg.solve(L, Y)
#         except RuntimeError:
#             # 如果求解失败，使用伪逆作为后备
#             W = torch.linalg.pinv(L) @ Y
        
#         return W[:n], W[n:]
    
#     def forward(self, x):
#         device = x.device
#         B = x.size(0)
        
#         # 提取特征
#         features = self.feature_extractor(x).view(B, -1)
        
#         # 预测控制点位移（确保梯度连接）
#         displacement = self.fc(features).view(B, self.num_control_points, 2)
#         target_points = self.control_points.unsqueeze(0) + displacement
        
#         # 批量计算TPS参数（保持梯度）
#         W, A = [], []
#         for b in range(B):
#             W_b, A_b = self._compute_tps_weights(
#                 self.control_points.to(device),
#                 target_points[b]
#             )
#             W.append(W_b)
#             A.append(A_b)
#         W = torch.stack(W)  # [B, num_ctrl, 2]
#         A = torch.stack(A)  # [B, 3, 2]
        
#         # 生成采样网格
#         identity = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
#         grid = F.affine_grid(identity, x.size(), align_corners=True)
        
#         # 应用变换
#         warped_grid = self._apply_tps_transform(grid, W, A)
        
#         return F.grid_sample(x, warped_grid, align_corners=True)
class TPSTransformer(nn.Module):
    def __init__(self, in_channels=1, num_control_points=9):
        super().__init__()
        self.num_control_points = num_control_points
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 预测控制点位移
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * num_control_points)
        )
        
        # 注册控制点和辅助张量为buffer
        self.register_buffer('control_points', self._init_control_points())
        self.register_buffer('zero_3x3', torch.zeros(3, 3))
        self.register_buffer('zero_3x2', torch.zeros(3, 2))
        self.register_buffer('ones_col', torch.ones(num_control_points, 1))
    
    def _init_control_points(self):
        """初始化均匀分布的控制点网格"""
        ctrl_pts = torch.zeros(self.num_control_points, 2)
        for i in range(self.num_control_points):
            x = (i % int(math.sqrt(self.num_control_points))) / (math.sqrt(self.num_control_points)-1) * 2 - 1
            y = (i // int(math.sqrt(self.num_control_points))) / (math.sqrt(self.num_control_points)-1) * 2 - 1
            ctrl_pts[i, 0] = x
            ctrl_pts[i, 1] = y
        return ctrl_pts
    
    def _compute_tps_weights(self, source, target):
        """计算TPS变换参数（完全设备感知版本）"""
        device = source.device
        
        # 计算距离矩阵
        K = torch.zeros(self.num_control_points, self.num_control_points, device=device)
        for i in range(self.num_control_points):
            for j in range(self.num_control_points):
                diff = source[i] - source[j]
                r2 = torch.dot(diff, diff)
                K[i,j] = r2 * torch.log(r2) if r2 > 0 else 0
        
        # 构造线性方程组（使用预注册的buffer）
        P = torch.cat([self.ones_col.to(device), source], dim=1)
        L_upper = torch.cat([K, P], dim=1)
        L_lower = torch.cat([P.T, self.zero_3x3.to(device)], dim=1)
        L = torch.cat([L_upper, L_lower], dim=0)
        
        # 解方程组
        Y = torch.cat([target, self.zero_3x2.to(device)], dim=0)
        W = torch.linalg.solve(L, Y)
        
        return W[:self.num_control_points], W[self.num_control_points:]
    
    def _apply_tps_transform(self, grid, W, A):
        """完全修正的TPS变换应用方法"""
        batch_size, H, W_dim, _ = grid.shape
        device = grid.device
        num_points = H * W_dim
    
        # 1. 准备控制点和网格点
        points = grid.reshape(batch_size, -1, 2)  # [B, H*W, 2]
        control_points = self.control_points.to(device)  # [num_ctrl, 2]
    
        # 2. 计算径向基函数U [B, H*W, num_ctrl]
        diffs = points.unsqueeze(2) - control_points.unsqueeze(0).unsqueeze(0)  # [B, H*W, num_ctrl, 2]
        r2 = (diffs ** 2).sum(dim=-1)  # [B, H*W, num_ctrl]
        U = torch.where(r2 > 0, r2 * torch.log(r2), torch.zeros_like(r2))
    
        # 3. 计算弯曲部分 [B, H*W, 2]
        warp = torch.bmm(U, W)  # 使用批量矩阵乘法
    
        # 4. 计算仿射部分 [B, H*W, 2]
        affine = A[:, 0:1] + (A[:, 1:2] * points[..., 0:1]) + (A[:, 2:3] * points[..., 1:2])
    
        # 5. 合并结果
        transformed = affine + warp
    
        # 6. 恢复原始形状 [B, H, W, 2]
        return transformed.reshape(batch_size, H, W_dim, 2)
    
    def forward(self, x):
        device = x.device
        B = x.size(0)
        
        # 提取特征
        features = self.feature_extractor(x).view(B, -1)
        
        # 预测控制点位移
        displacement = self.fc(features).view(B, self.num_control_points, 2)
        target_points = self.control_points.unsqueeze(0).to(device) + displacement
        
        # 批量计算TPS参数
        W, A = [], []
        for b in range(B):
            W_b, A_b = self._compute_tps_weights(
                self.control_points.to(device),
                target_points[b]
            )
            W.append(W_b)
            A.append(A_b)
        W = torch.stack(W)  # [B, num_ctrl, 2]
        A = torch.stack(A)  # [B, 3, 2]
        
        # 生成采样网格
        grid = F.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1),
            x.size(),
            align_corners=True
        )
        
        # 应用变换
        warped_grid = self._apply_tps_transform(grid, W, A)
        
        return F.grid_sample(x, warped_grid, align_corners=True)        
class SpatialTransformer(nn.Module):
    """空间变换网络（自动适配输入尺寸）"""
    def __init__(self, in_channels=1):
        super().__init__()
        # 定位网络（输出固定尺寸特征图）
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # 动态计算全连接层输入尺寸
        self.fc_input_dim = 1960
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # 初始化变换矩阵
        self._init_weights()

    # def _get_fc_input_dim(self, in_channels):
    #     """通过前向传播计算实际特征维度"""
    #     test_input = torch.randn(1, in_channels, 224, 224)  # 假设输入为224x224
    #     test_output = self.localization(test_input)
    #     print(test_output.view(-1).shape[0])
    #     return test_output.view(-1).shape[0]

    def _init_weights(self):
        """初始化仿射变换参数"""
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        # 特征提取
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)  # 展平
        
        # 预测变换参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # 应用空间变换
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


