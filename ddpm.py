#%%
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from torchvision.utils import make_grid
import logging

import random
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
#%%

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Cataract"
    args.epochs = 500
    args.batch_size = 2
    args.image_size = 64
    args.dataset_path = r"images/CPS_Dog_Cataract_Rand"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    
def sample_images(dataset_path, output_path, grid_size=8):
    # 데이터셋 불러오기 및 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])
    # 데이터셋 불러오기
    dataset = dset.ImageFolder(dataset_path, transform=transform)
    
    # 랜덤으로 64개의 이미지 선택
    images = []
    for i in range(grid_size**2):
        index = random.randint(0, len(dataset) - 1)
        image, _ = dataset[index]
        images.append(image)
    
    # 이미지 그리드 생성
    grid = make_grid(images, nrow=grid_size, pad_value=1)

    # 이미지 저장
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(16, 16))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'sample.png'))

#두줄 샘플링
if __name__ == '__main__':
    launch()
    
    dataset_path = 'images/CPS_Dog_Cataract_Rand'
    output_path = 'Results/DDPM_Cataract'
    sample_images(dataset_path, output_path, grid_size=8)
    
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("models/DDPM_Uncondtional_FFHQ-3/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 8)
    print(x.shape)

    x_grid = make_grid(x, nrow=4, pad_value=1)
    x_grid_np = (x_grid.permute(1, 2, 0)*255).type(torch.uint8).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(x_grid_np)
    ax.set_title("Diffusion Sampling Results", fontsize=16)
    plt.show()

#원본데이터 샘플링 파일
    
""" 
if __name__ == '__main__':
    launch()
    
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("models\DDPM_Uncondtional\ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 8)
    print(x.shape)
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
         torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
 """
