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

def calc_schedules(betasss, ts, dev):
    global device
    global timesteps
    global betas
    global alphas
    global alphas_cumprod
    global alphas_cumprod_prev
    global sqrt_recip_alphas
    global sqrt_alphas_cumprod
    global sqrt_one_minus_alphas_cumprod
    global posterior_variance
    
    device = dev
    betas = betasss
    timesteps = ts
    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def custom_weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def map_interval(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max() - tensor.min())
    tensor = tensor*2 -1
    return tensor

def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu().type(torch.int64))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

''' forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image



#SAMPLING_IMGAES
#----------------------------------

@torch.no_grad()
def p_sample(model, x, t, t_index, batch_size):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, torch.zeros(batch_size, 12).to(device)) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

##New
def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

@torch.no_grad()
def p_sample_new(self, x, t: int, x_self_cond = None):
    b, *_, device = *x.shape, x.device
    batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
    model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
    noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, batch_size):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, batch_size)
        #imgs.append(img.cpu().numpy())
        imgs.append(img)
    return imgs

@torch.no_grad()
def p_sample_loop_new(self, shape, return_all_timesteps = False):
    batch, device = shape[0], self.betas.device

    img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, self_cond)
        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    return ret

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), batch_size=batch_size)'''


 #NOISE SCHEDULES
 #--------------------------------------------------------
    
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


#LOSSES
#---------------------------------------------------------
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1", augm=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, augm)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses_new(self, x_start, t, noise = None):
    b, c, h, w = x_start.shape
    noise = default(noise, lambda: torch.randn_like(x_start))

    # noise sample

    x = self.q_sample(x_start = x_start, t = t, noise = noise)

    # if doing self-conditioning, 50% of the time, predict x_start from current set of times
    # and condition with unet with that
    # this technique will slow down training by 25%, but seems to lower FID significantly

    x_self_cond = None
    if self.self_condition and random() < 0.5:
        with torch.no_grad():
            x_self_cond = self.model_predictions(x, t).pred_x_start
            x_self_cond.detach_()

    # predict and take gradient step

    model_out = self.model(x, t, x_self_cond)

    if self.objective == 'pred_noise':
        target = noise
    elif self.objective == 'pred_x0':
        target = x_start
    elif self.objective == 'pred_v':
        v = self.predict_v(x_start, t, noise)
        target = v
    else:
        raise ValueError(f'unknown objective {self.objective}')

    loss = self.loss_fn(model_out, target, reduction = 'none')
    loss = reduce(loss, 'b ... -> b (...)', 'mean')

    loss = loss * extract(self.loss_weight, t, loss.shape)
    return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_data(path, batch_size, image_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size,image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Lambda(lambda t: (t * 2) - 1)
    ])
    
    g = torch.Generator()
    g.manual_seed(42)
    dataset = torchvision.datasets.ImageFolder(path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g,drop_last=False)
    return dataloader