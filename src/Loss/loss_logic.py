import torch
from torchvision import models
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initializing a pre-trained VGG model for perceptual loss
vgg = models.vgg16(pretrained=True).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False



def vae_gaussian_kl_loss(mu, logvar):
        # Inspired by Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return KLD.mean()



def perceptual_loss(x_reconstructed, x, vgg):
    # Extracting features from certain VGG layers for perceptual loss
    def vgg_features(x):
        # Passing input through VGG layers up to the relu3_3 layer
        return vgg[:16](x)
    
    x_recon_features = vgg_features(x_reconstructed)
    x_features = vgg_features(x)
    return F.mse_loss(x_recon_features, x_features)



def reconstruction_loss(x_reconstructed, x):
    # here using BCE loss for more stable color reproduction if pixel values are in [0, 1]
    x = (x + 1) / 2
    x_reconstructed = (x_reconstructed + 1) / 2
    if x.min() < 0 or x.max() > 1:
        print(x)
        print(f"Normalizeing {x.min().item()} and {x.max().item()}")

    if x_reconstructed.min() < 0 or x_reconstructed.max() > 1:
        print(f"Normalizeing {x_reconstructed.min().item()} and {x_reconstructed.max().item()}")

    
    recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='mean') / x.size(0)
    return recon_loss



def vae_loss(y_pred, y_true, epoch):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    
    # applying perceptual loss to encourage color and feature realism
    percept_loss = perceptual_loss(recon_x, y_true, vgg)
    
    # Combining perceptual and reconstruction loss
    combined_recon_loss = recon_loss + 0.1 * percept_loss
    
    # KL Divergence loss with dynamic weighting
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    kl_weight = min(1.0, epoch / 20)  # Gradually increase KL weight over first 20 epochs
    
    # final weighted loss
    return 500 * combined_recon_loss + kld_loss * kl_weight

