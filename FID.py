#%%
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm

#%%
def load_model_state(model, filename):
    model_info = torch.load(f"saved_models/{filename}.pt")
    model.load_state_dict(model_info["model_state_dict"])
    return model


# 배치 크기
batch_size = 128
# 이미지의 크기입니다. 모든 이미지들은 transformer를 이용해 64로 크기가 통일됩니다.
image_size = 64
# 이미지의 채널 수로, RGB 이미지이기 때문에 3으로 설정합니다.
nc = 3
# 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
nz = 100
# 생성자를 통과하는 특징 데이터들의 채널 크기
ngf = 64
# 구분자를 통과하는 특징 데이터들의 채널 크기
ndf = 64
# 학습할 에폭 수
num_epochs = 100
# 옵티마이저의 학습률
lr = 0.0002
# Adam 옵티마이저의 beta1 하이퍼파라미터
beta1 = 0.5

# 사용가능한 gpu 번호. CPU를 사용해야 하는경우 0으로 설정하세요
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device,"is available!")

class GeneratorModel(nn.Module):
    def __init__(self, n_z, n_fmps, n_c):
        super(GeneratorModel, self).__init__()
        self.n_z = n_z
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, n_fmps*8, 4, 1, 0),
            nn.BatchNorm2d(n_fmps*8),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(n_fmps*8, n_fmps*4, 4, 2, 1),
            nn.BatchNorm2d(n_fmps*4),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(n_fmps*4, n_fmps*2, 4, 2, 1),
            nn.BatchNorm2d(n_fmps*2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(n_fmps*2, n_fmps*1, 4, 2, 1),
            nn.BatchNorm2d(n_fmps*1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(n_fmps, n_c, 4, 2, 1),
            nn.Tanh()
        )
        #self.net.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, self.n_z, 1, 1)
        x = self.net(x)
        return x

# 모델을 로드합니다
netG = GeneratorModel(nz, ngf, nc).to(device)
netG = load_model_state(netG, 'Generator300')
netG.to(device)
netG.eval()  # 모델을 평가 모드로 설정합니다. 이렇게 하면 dropout이나 batchnorm 같은 레이어들이 학습 모드에서와는 다르게 동작합니다.

with torch.no_grad():
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    fake_images = netG(fixed_noise)
    save_image(fake_images, 'fake_images.png')

# 1. fake_image 디렉토리 생성
if not os.path.exists('fake_image_Dog2/fake_image'):
    os.makedirs('fake_image_Dog2/fake_image')

# 2. 가짜 이미지 생성 및 fake_image 디렉토리에 저장
required_batches = 6000 // batch_size  # 필요한 배치 수 계산
for batch_num in range(required_batches):
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise)
    for image_num, image in enumerate(fake):
        vutils.save_image(image, f'fake_image_Dog2/fake_image/fake_{batch_num * batch_size + image_num}.png', normalize=True)
        

# Inception 모델 로딩
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

def get_features(data_loader, model):
    all_features = []
    for images, _ in tqdm(data_loader, desc="Extracting Features"):  # tqdm 추가
        with torch.no_grad():
            features = model(images.to(device))
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

# FID 계산
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 1. 실제 데이터셋에 대한 Dataloader 생성
transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

real_dataset = dset.ImageFolder(root='images/CPS_Dog_Cataract_Rand', transform=transform)
real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

# 2. 가짜 데이터셋에 대한 Dataloader 생성
fake_dataset = dset.ImageFolder(root = 'fake_image_Dog2', transform=transform)
fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

# 몇 개의 샘플 이미지를 시각화합니다.
sample_batch = next(iter(real_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Images")
plt.imshow(np.transpose(vutils.make_grid(sample_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

#%%

# 실제 데이터와 생성된 데이터의 특징 벡터를 추출
print("Extracting features from real data...")
real_features = get_features(real_dataloader, inception_model)
print("Extracting features from fake data...")
fake_features = get_features(fake_dataloader, inception_model)

# FID 계산
print("Calculating FID...")
fid_value = calculate_fid(real_features, fake_features)
print("FID:", fid_value)

# %%
