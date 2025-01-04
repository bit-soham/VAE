import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.attention import SelfAttention
from src.vae_attentionblock import VAE_AttentionBlock
from src.vae_residualblock import VAE_ResidualBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__()

        # Define the layers individually
        self.initial_conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(4, 512, kernel_size=3, padding=1)
        self.res_block1 = VAE_ResidualBlock(512, 512).to(device)
        self.attention_block = VAE_AttentionBlock(512).to(device)
        self.res_block2 = VAE_ResidualBlock(512, 512).to(device)
        self.res_block3 = VAE_ResidualBlock(512, 512).to(device)
        self.res_block4 = VAE_ResidualBlock(512, 512).to(device)
        self.res_block5 = VAE_ResidualBlock(512, 512).to(device)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.res_block6 = VAE_ResidualBlock(512, 512).to(device)
        self.res_block7 = VAE_ResidualBlock(512, 512).to(device)
        self.res_block8 = VAE_ResidualBlock(512, 512).to(device)

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.res_block9 = VAE_ResidualBlock(512, 256).to(device)

        self.res_block10 = VAE_ResidualBlock(256, 256).to(device)
        self.res_block11 = VAE_ResidualBlock(256, 256).to(device)

        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.res_block12 = VAE_ResidualBlock(256, 128).to(device)
        self.res_block13 = VAE_ResidualBlock(128, 128).to(device)
        self.res_block14 = VAE_ResidualBlock(128, 128).to(device)

        self.group_norm = nn.GroupNorm(32, 128)
        self.silu = nn.SiLU()
        self.output_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)


    def forward(self, x):
        # Scaling added by the Encoder is removed here
        x /= 0.18215

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.initial_conv(x)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.conv1(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.res_block1(x)
         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.attention_block(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.res_block2(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.res_block3(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.res_block4(x)
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        x = self.res_block5(x)

        # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
        # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.upsample1(x)
        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.conv2(x)
         # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.res_block6(x)
         # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.res_block7(x)
         # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        x = self.res_block8(x)

        # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
        x = self.upsample2(x)
        # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
        x = self.conv3(x)
        # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.res_block9(x)
         # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.res_block10(x)
         # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        x = self.res_block11(x)
        # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
        x = self.upsample3(x)
        # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
        x = self.conv4(x)
        # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.res_block12(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.res_block13(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.res_block14(x)
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.group_norm(x)

        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        x = self.silu(x)
        
        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
        x = self.output_conv(x)



#         # # (Batch_Size, 128, Height, Width) -> (Batch_Size, 6, Height, Width)
#         x = self.output_conv(x)

        # (Batch_Size, 6, Height , Width ) -> two tensors of shape (Batch_Size, 6, Height , Width )
#         mean, log_variance = torch.chunk(x, 2, dim=1)

#          # Clamp the log variance
#         log_variance = torch.clamp(log_variance, -30, 20)
#         # (Batch_Size, 3, Height , Width )
#         x = mean  # Set `x` to the mean for the final reconstruction

#         return {"x": x, "mu": mean, "log_var": log_variance}

        x = torch.tanh(x) 

        return x