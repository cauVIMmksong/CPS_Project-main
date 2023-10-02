import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import matplotlib.pyplot as plt


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)  # 이미지를 [-1, 1] 범위로 클램핑
        return x



def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(4).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional_CIFAR10(v2)"
    args.epochs = 100
    args.batch_size = 16
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"images/cifar10-32/train"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

def save_generated_images(model, diffusion, num_images, save_dir, image_size=64, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(num_images), position=0, desc="Generating images"):
        y = torch.randint(0, 10, (1,)).long().to(device)  # 무작위 클래스 선택
        x = diffusion.sample(model, 1, y, cfg_scale=0)  # 이미지 1장 생성
        
        x = (x + 1) / 2  # [-1, 1] 범위의 텐서를 [0, 1] 범위의 텐서로 변환
        #x = (x * 255).type(torch.uint8)  # [0, 1] 범위의 텐서를 [0, 255] 범위의 uint8 텐서로 변환
        
        save_path = os.path.join(save_dir, f"sample_{i}.jpg")
        save_image(x[0], save_path)

        # 메모리 정리
        del x
        torch.cuda.empty_cache()

'''
if __name__ == '__main__':
    launch()

    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./models/DDPM_conditional_CIFAR10/ckpt.pt")
    model.load_state_dict(ckpt)
    model.eval()  # 생성 모드로 전환
    diffusion = Diffusion(img_size=64, device=device)

    save_dir = "CIFAR10_fake/img"
    save_generated_images(model, diffusion, 6000, save_dir)
'''


# 기존 샘플링 코드
if __name__ == '__main__':
    launch()
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./models/DDPM_conditional_CIFAR10v2/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    n = 16
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=0)
    plot_images(x)
    # 이미지 저장
    plt.savefig('results/DDPM_conditional_CIFAR10/sampled_images5.png', bbox_inches='tight', pad_inches=0)

'''
if __name__ == '__main__':
    launch()
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./models/DDPM_conditional_CIFAR10/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    num_classes = 10  # 클래스의 총 개수
    n_per_class = 1  # 각 클래스당 생성할 이미지 개수
    images = []
    for class_label in range(num_classes):
        y = torch.Tensor([class_label] * n_per_class).long().to(device)
        x = diffusion.sample(model, n_per_class, y, cfg_scale=0)
        images.append(x)
    images = torch.cat(images)
    images_cpu = images.cpu().numpy()  # 이미지를 CPU로 이동 후 NumPy 배열로 변환

     # 이미지를 1x10 그리드로 배치하여 출력
    plt.figure(figsize=(10, 1))  # 가로 크기를 크게 조정하여 가로로 길게 배치
    grid = np.reshape(images_cpu, (num_classes * n_per_class, 3, 64, 64))
    grid = np.transpose(grid, (0, 2, 3, 1))
    grid = np.concatenate(grid, axis=1)
    grid = np.concatenate(grid, axis=0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
'''

    