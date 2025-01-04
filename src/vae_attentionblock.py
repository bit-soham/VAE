import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
      super().__init__()
      self.groupnorm = nn.GroupNorm(32, channels)
      self.attention = SelfAttention(1, channels)


    def forward(self, x):

      # x: (Batch_Size, Features, Height, Width)
      residue = x

      # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
      x = self.groupnorm(x)

      n, c, h, w = x.shape    # n is the batch size, c is the no. of the features, height, weight

      # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
      x = x.view((n, c, h * w))

      # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
      x = x.transpose(-1, -2)

      # Perform self-attention WITHOUT mask on the pixels of the image
      # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
      x = self.attention(x)

      # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
      x = x.transpose(-1, -2)

      # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
      x = x.view((n, c, h, w))

      # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
      x += residue

      # (Batch_Size, Features, Height, Width)
      return x
