import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.vae_model import VanillaVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model checkpoint
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return model, start_epoch

# Image transform
transform = transforms.Compose([
    transforms.Resize((525, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Using 3 channels
])

def load_image(image_path):
    """Load and transform image."""
    image = Image.open(image_path)
    image_tensor = transform(image).to(device)
    return image_tensor

def show_image(image_tensor):
    """Display image from tensor."""
    image = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    image = (image + 1) / 2  # Scale to [0, 1]
    image = np.clip(image, 0, 1)  # Clip to ensure no values are out of range
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def evaluate_mae_mse(original, reconstructed):
    """Calculate and return MAE and MSE."""
    mae = torch.mean(torch.abs(original - reconstructed))
    mse = torch.mean((original - reconstructed) ** 2)
    return mae.item(), mse.item()

def main():
    # Initializing the VAE model
    vae = VanillaVAE(in_channels=3, latent_dim=128, num_epochs=600).to(device)

    # Loading the checkpoint
    vae, start_epoch = load_checkpoint(vae, '..\models\vae_model_epoch_600.pth')
    vae.eval()  # Set to evaluation mode

    # Loading and transform image
    image_path = '../data/image_to_test.jpg' 
    image_tensor = load_image(image_path)

    # Passing through VAE to get the reconstructed image
    _, _, reconstructed = vae(image_tensor.unsqueeze(0))

    # the original and reconstructed images
    show_image(image_tensor)
    show_image(reconstructed.squeeze(0))

    # Calculating and display MAE and MSE
    mae, mse = evaluate_mae_mse(image_tensor, reconstructed)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

if __name__ == "__main__":
    main()
