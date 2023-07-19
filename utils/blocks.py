import math
from inspect import isfunction
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from PIL import Image
import requests
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from torch.utils.data import DataLoader
import time
import pandas as pd

import dnnlib
import torchvision
from torchvision.utils import save_image

import cv2

from utils.losses_samples import *



#ATTENTION LAYERS
#-------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class EfficientAttention(nn.Module):
    """https://github.com/cmsflash/efficient-attention"""
    "Efficient attention module that, however, did not improve the performance in our case. We stick with the linear attention"
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
def plot_images2(images):
    plt.figure(figsize=(32, 32))
    vec = torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu()
    for i in range (vec.shape[0]):
        ax = fig.add_subplot(math.ceil(n/5),5,i+1)
        plt.imshow(vec[i], cmap=plt.cm.gray)
    plt.show()
    
def plot_images(pixel_arrays, n=100):
    "Function to plot an array of images"
    if torch.is_tensor(pixel_arrays):
        pixel_arrays = cv2.normalize(src=pixel_arrays.cpu().detach().numpy().transpose(0,2,3,1), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(pixel_arrays) < n:
        n = len(pixel_arrays)
    fig = plt.figure(figsize=(15,int(n*0.5)))
    img_list = list()
    img_file_list = list()
    mask_list = list()
    for i in range (n):
        ax = fig.add_subplot(math.ceil(n/5),5,i+1)
        plt.imshow(pixel_arrays[i])

    plt.show()


#RESNET and CONVENEXT blocks
#----------------------------------------------------
class Block(nn.Module):
    "Simple block with convolution, groupnorm and SiLU acivation function"
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.mlp2 = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, augm_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
            
        if exists(self.mlp2) and exists(augm_emb):
            condition = self.mlp2(augm_emb.squeeze(1))
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)
    
class SEBlock(nn.Module):
    "https://arxiv.org/abs/2101.04775"
    def __init__(self, ch_in, ch_out, emb_dim=256):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                ch_out
            ),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)
    
class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

