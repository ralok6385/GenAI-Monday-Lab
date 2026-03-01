import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data Preprocessing
# Normalize images to [-1, 1]
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# -----------------------------
# Encoder–Decoder CNN
# -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AutoEncoder().to(device)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 5

for epoch in range(epochs):
    for images, _ in train_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# Visualization
# -----------------------------
dataiter = iter(train_loader)
images, _ = next(dataiter)
images = images.to(device)

with torch.no_grad():
    reconstructed = model(images)

# Denormalize
images = images * 0.5 + 0.5
reconstructed = reconstructed * 0.5 + 0.5

# Plot
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Row titles (this is the key fix)
for ax in axes[0]:
    ax.set_title("Original", fontsize=12)

for ax in axes[1]:
    ax.set_title("Reconstructed", fontsize=12)

for i in range(8):
    axes[0, i].imshow(images[i].permute(1, 2, 0).cpu())
    axes[0, i].axis("off")

    axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu())
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("result.png")
plt.show()