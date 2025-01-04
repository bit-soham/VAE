import math
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.Loss.loss_logic import vae_loss
from src.decoder import VAE_Decoder
from src.encoder import VAE_Encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VanillaVAE(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List[int] = None, num_epochs: int = 25):
        """
        Initialize the VAE model with encoder and decoder architectures.
        """
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.encoder = VAE_Encoder().to(device)
        self.decoder = VAE_Decoder().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.num_epochs = num_epochs
    
    def getEncoder(self, nk, x):
        nx, _, _ = self.encoder(x)
        return nx
        
    def forward(self, x):
        # passing the input through the encoder to get the latent vector
        z, z_mean, z_log_var = self.encoder(x)
        # passing the latent vector through the decoder to get the reconstructed image
        reconstruction = self.decoder(z)
        # returning the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:  # lv is log_variance
        # print(f"mu-", mu.size())
        # print("lv-", lv.size())
        # print("eps-", epsilon.size())
        epsilon = torch.randn(mu.shape, device=mu.device)
        sigma = torch.exp(0.5 * lv)  # Standard deviation
        z = mu + sigma * epsilon
        return z


    def train(self, train_loader, start_epoch=0) -> List[torch.Tensor]:
        self.to(device)

        for epoch in range(start_epoch, self.num_epochs):  
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                pred = self(batch)
                loss = vae_loss(pred, batch, epoch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()  # Accumulating loss for average calculation
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

                # Saving model and optimizer state at epoch 40 and every 49 epochs
                if (epoch + 1) == 40 or (epoch + 1) % 50 == 0:
                    torch.save({
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }, f"vae_model_epoch_{epoch + 1}.pth")
                    print(f"Model and optimizer state saved at epoch {epoch + 1}")
                    
                if loss == 0:
                    print("stopped")
                    break

            print(f"Epoch: {epoch + 1}, Avg_Loss: {epoch_loss / len(train_loader)}")
            print()

        return loss


