import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Metric

# Hyperparameters
dataset_name = "FFHQ"
dataset_repetitions = 5
num_epochs = 1
image_size = 64
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20
min_signal_rate = 0.02
max_signal_rate = 0.95
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

# Data pipeline
def preprocess_image(data):
    height = data["image"].shape[0]
    width = data["image"].shape[1]
    crop_size = min(height, width)
    image = data["image"][(height - crop_size) // 2 : (height + crop_size) // 2, (width - crop_size) // 2 : (width + crop_size) // 2]
    image = transforms.functional.resize(image, (image_size, image_size), antialias=True)
    image = transforms.functional.to_tensor(image)
    return transforms.functional.clip(image / 255.0, 0.0, 1.0)

def prepare_dataset(split):
    dataset = datasets.load_dataset(dataset_name, split=split, shuffle_files=True)
    dataset = dataset.map(preprocess_image, num_parallel_calls=torch.utils.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat(dataset_repetitions)
    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.batch(batch_size, drop_last=True)
    dataset = dataset.prefetch(buffer_size=torch.utils.data.AUTOTUNE)
    return dataset

train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

# KID Metric
class KID(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("kid_tracker", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, real_images, generated_images):
        real_features = inception_encoder(real_images)
        generated_features = inception_encoder(generated_images)
        kernel_real = polynomial_kernel(real_features, real_features)
        kernel_generated = polynomial_kernel(generated_features, generated_features)
        kernel_cross = polynomial_kernel(real_features, generated_features)
        mean_kernel_real = (torch.sum(kernel_real) - torch.sum(torch.eye(kernel_real.size(0)) * kernel_real)) / (kernel_real.size(0) * (kernel_real.size(0) - 1))
        mean_kernel_generated = (torch.sum(kernel_generated) - torch.sum(torch.eye(kernel_generated.size(0)) * kernel_generated)) / (kernel_generated.size(0) * (kernel_generated.size(0) - 1))
        mean_kernel_cross = torch.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        self.kid_tracker += kid

    def compute(self):
        return self.kid_tracker

def polynomial_kernel(features_1, features_2):
    feature_dimensions = features_1.size(1)
    return torch.pow((features_1 @ torch.transpose(features_2, 0, 1) / feature_dimensions + 1.0), 3.0)

class InceptionEncoder(nn.Module):
    def __init__(self):
        super(InceptionEncoder, self).__init__()
        self.model = models.inception_v3(pretrained=True, aux_logits=False, transform_input=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

class DiffusionModel(pl.LightningModule):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.inception_encoder = InceptionEncoder()
        self.network = self.get_network()

        self.normalizer = nn.Identity()
        self.ema_network = self.network.copy()

        self.kid = KID()

    def get_network(self):
        noisy_images = nn.Identity()
        noise_variances = nn.Identity()

        e = self.sinusoidal_embedding(noise_variances)
        e = nn.Upsample(size=image_size, mode="nearest")(e)

        x = nn.Conv2d(3, widths[0], kernel_size=1)(noisy_images)
        x = torch.cat([x, e], dim=1)

        skips = []
        for width in widths[:-1]:
            x = self.down_block(x, skips, width)

        for _ in range(block_depth):
            x = self.residual_block(x, widths[-1])

        for width in reversed(widths[:-1]):
            x = self.up_block(x, skips, width)

        x = nn.Conv2d(widths[0], 3, kernel_size=1, bias=False)(x)

        return nn.Sequential(noisy_images, noise_variances, x)

    def down_block(self, x, skips, width):
        for _ in range(block_depth):
            residual = x if x.size(1) == width else nn.Conv2d(x.size(1), width, kernel_size=1)(x)
            x = nn.BatchNorm2d(x.size(1), affine=False)(x)
            x = nn.Conv2d(x.size(1), width, kernel_size=3, padding=1)(x)
            x = nn.ReLU(inplace=True)(x)
            x = nn.Conv2d(width, width, kernel_size=3, padding=1)(x)
            x = x + residual
        skips.append(x)
        x = nn.AvgPool2d(kernel_size=2)(x)
        return x

    def up_block(self, x, skips, width):
        x = nn.Upsample(scale_factor=2, mode="bilinear")(x)
        x = torch.cat([x, skips.pop()], dim=1)
        for _ in range(block_depth):
            residual = x if x.size(1) == width else nn.Conv2d(x.size(1), width, kernel_size=1)(x)
            x = nn.BatchNorm2d(x.size(1), affine=False)(x)
            x = nn.Conv2d(x.size(1), width, kernel_size=3, padding=1)(x)
            x = nn.ReLU(inplace=True)(x)
            x = nn.Conv2d(width, width, kernel_size=3, padding=1)(x)
            x = x + residual
        return x

    def sinusoidal_embedding(self, x):
        embedding_min_frequency = 1.0
        frequencies = torch.exp(torch.linspace(
            torch.log(torch.tensor(embedding_min_frequency)),
            torch.log(torch.tensor(embedding_max_frequency)),
            embedding_dims // 2
        ))
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = torch.cat([torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=1)
        return embeddings

    def residual_block(self, x, width):
        residual = x if x.size(1) == width else nn.Conv2d(x.size(1), width, kernel_size=1)(x)
        x = nn.BatchNorm2d(x.size(1), affine=False)(x)
        x = nn.Conv2d(x.size(1), width, kernel_size=3, padding=1)(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Conv2d(width, width, kernel_size=3, padding=1)(x)
        x = x + residual
        return x

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.std
        return torch.clamp(images, 0.0, 1.0)

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        next_noisy_images = initial_noise
        step_size = 1.0 / diffusion_steps

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = torch.ones((noisy_images.size(0), 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps):
        initial_noise = torch.randn(num_images, image_size, image_size, 3)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def diffusion_schedule(self, diffusion_times):
        start_angle = torch.acos(torch.tensor(max_signal_rate))
        end_angle = torch.acos(torch.tensor(min_signal_rate))
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises, pred_images = network([noisy_images, noise_rates**2])
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images = self.normalizer(batch, training=True)
        noises = torch.randn(batch_size, image_size, image_size, 3)

        diffusion_times = torch.rand(batch_size, 1, 1, 1)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)

        noise_loss = F.l1_loss(noises, pred_noises)
        image_loss = F.l1_loss(images, pred_images)

        self.log("n_loss", noise_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("i_loss", image_loss, prog_bar=True, on_step=False, on_epoch=True)

        for weight, ema_weight in zip(self.network.parameters(), self.ema_network.parameters()):
            ema_weight.data.mul_(ema).add_((1 - ema) * weight.data)

        return noise_loss

    def validation_step(self, batch, batch_idx):
        images = self.normalizer(batch, training=False)
        noises = torch.randn(batch_size, image_size, image_size, 3)

        diffusion_times = torch.rand(batch_size, 1, 1, 1)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)

        noise_loss = F.l1_loss(noises, pred_noises)
        image_loss = F.l1_loss(images, pred_images)

        self.log("i_loss", image_loss, prog_bar=True, on_step=False, on_epoch=True)

        images = self.denormalize(images)
        generated_images = self.generate(num_images=batch_size, diffusion_steps=kid_diffusion_steps)
        self.kid.update(images, generated_images)

        return {
            "val_kid": self.kid.compute(),
            "val_i_loss": image_loss,
            "val_n_loss": noise_loss,
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer

model = DiffusionModel()

# Checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor="val_kid",
    mode="min",
    dirpath="checkpoints/",
    filename="diffusion_model-{epoch:02d}-{val_kid:.4f}",
    save_weights_only=True,
    save_top_k=1,
)

# Training
trainer = pl.Trainer(
    max_epochs=num_epochs,
    gpus=1 if torch.cuda.is_available() else None,
    progress_bar_refresh_rate=20,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataset, val_dataset)

# Generate images using the best model
model = DiffusionModel.load_from_checkpoint(checkpoint_callback.best_model_path)
model.eval()
generated_images = model.generate(num_images=3 * 6, diffusion_steps=plot_diffusion_steps)
generated_images = model.denormalize(generated_images)

import matplotlib.pyplot as plt

plt.figure(figsize=(6 * 2.0, 3 * 2.0))
for row in range(3):
    for col in range(6):
        index = row * 6 + col
        plt.subplot(3, 6, index + 1)
        plt.imshow(generated_images[index].permute(1, 2, 0))
        plt.axis("off")
plt.tight_layout()
plt.show()
