import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

print("Starting Neural Style Transfer...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content = load_image("content.jpg")
style = load_image("style.jpg")

print("Images loaded")

vgg = models.vgg19(pretrained=True).features.to(device).eval()

for param in vgg.parameters():
    param.requires_grad = False

generated = content.clone().requires_grad_(True).to(device)

optimizer = torch.optim.Adam([generated], lr=0.01)

print("Model ready. Running optimization...")

for step in range(50):

    gen_features = vgg(generated)
    content_features = vgg(content)
    style_features = vgg(style)

    loss = torch.mean((gen_features - content_features)**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Step:", step, "Loss:", loss.item())

output = generated.clone().detach().cpu().squeeze()
output = transforms.ToPILImage()(output)
output.save("stylized_output.png")

print("Saved stylized_output.png")