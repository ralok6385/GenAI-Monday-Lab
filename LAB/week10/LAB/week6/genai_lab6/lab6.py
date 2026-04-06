import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset (Edges → Image)
# -----------------------------
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

def edge_detect(img):
    img = np.array(img)
    edges = cv2.Canny(img, 100, 200)
    edges = np.stack([edges] * 3, axis=-1)
    edges = edges.astype(np.float32) / 255.0   # IMPORTANT FIX
    return TF.to_tensor(edges)

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        edge = edge_detect(img).float()
        img = TF.to_tensor(img).float()

        edge = (edge * 2) - 1
        img = (img * 2) - 1

        return edge, img

train_data = PairedDataset(dataset)
loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

# -----------------------------
# U-Net Generator
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        d1 = torch.relu(self.down1(x))
        d2 = torch.relu(self.down2(d1))
        u1 = torch.relu(self.up1(d2))
        u2 = torch.tanh(self.up2(torch.cat([u1, d1], 1)))
        return u2

# -----------------------------
# PatchGAN Discriminator
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

# -----------------------------
# Loss & Optimizers
# -----------------------------
adv_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# -----------------------------
# Training
# -----------------------------
epochs = 3

for epoch in range(epochs):
    for edge, real in loader:
        edge, real = edge.to(device), real.to(device)

        # Train Discriminator
        fake = G(edge)
        D_real = D(edge, real)
        D_fake = D(edge, fake.detach())

        loss_D = adv_loss(D_real, torch.ones_like(D_real)) + \
                 adv_loss(D_fake, torch.zeros_like(D_fake))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        D_fake = D(edge, fake)
        loss_G = adv_loss(D_fake, torch.ones_like(D_fake)) + l1_loss(fake, real)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | G Loss: {loss_G.item():.4f}")

# -----------------------------
# Visualization (Pix2Pix Style)
# -----------------------------
edge, real = next(iter(loader))
edge, real = edge.to(device), real.to(device)

with torch.no_grad():
    fake = G(edge)

def denorm(x):
    return (x + 1) / 2

edge_img = denorm(edge[0]).permute(1, 2, 0).cpu()
real_img = denorm(real[0]).permute(1, 2, 0).cpu()
fake_img = denorm(fake[0]).permute(1, 2, 0).cpu()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(edge_img)
ax[0].set_title("Input (Edges)")
ax[0].axis("off")

ax[1].imshow(real_img)
ax[1].set_title("Target (Real)")
ax[1].axis("off")

ax[2].imshow(fake_img)
ax[2].set_title("Generated")
ax[2].axis("off")

plt.tight_layout()
plt.savefig("result.png")
plt.show()