import torch
from dataset import get_dataloader
from src.vae_model import VanillaVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR ='path-to-the-imageDir-of-your-own-Dataset' ## for example -> '/kaggle/input/flickr8k/Images'
IMAGE_SIZE = 128

# Create data loaders
train_loader = get_dataloader(BASE_DIR, dataset_size=400, batch_size=16, image_size=IMAGE_SIZE)
test_loader = get_dataloader(BASE_DIR, dataset_size=180, batch_size=8, image_size=IMAGE_SIZE)


vae = VanillaVAE(in_channels=3, latent_dim=128, num_epochs=600).to(device)

vae.train(train_loader)