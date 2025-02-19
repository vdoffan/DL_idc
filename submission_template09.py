import torch
import torch.nn as nn

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок энкодера: conv -> relu -> max_pooling
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок декодера: conv -> relu -> upsample
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest')  # можно mode='bilinear' и align_corners=True при желании
    )
    return block

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.enc1_block = encoder_block(in_channels, 32, 7, 3)
        self.enc2_block = encoder_block(32, 64, 3, 1)
        self.enc3_block = encoder_block(64, 128, 3, 1)

        # dec1_block симметричен enc3_block: 128 -> 64
        self.dec1_block = decoder_block(128, 64, 3, 1)
        # dec2_block симметричен enc2_block, но на вход 64 (от dec1) + 64 (от enc2) = 128 каналов: 128 -> 32
        self.dec2_block = decoder_block(128, 32, 3, 1)
        # dec3_block симметричен enc1_block, но на вход 32 (от dec2) + 32 (от enc1) = 64 каналов: 64 -> out_channels
        self.dec3_block = decoder_block(64, out_channels, 3, 1)

    def forward(self, x):
        # downsampling part
        enc1 = self.enc1_block(x)  # out: 32 каналов
        enc2 = self.enc2_block(enc1)  # out: 64 каналов
        enc3 = self.enc3_block(enc2)  # out: 128 каналов

        # upsampling part
        dec1 = self.dec1_block(enc3)  # out: 64 каналов

        # skip connections
        dec2 = self.dec2_block(torch.cat([dec1, enc2], dim=1))  # 64 + 64 = 128 каналов -> out: 32 каналов
        dec3 = self.dec3_block(torch.cat([dec2, enc1], dim=1))  # 32 + 32 = 64 каналов -> out: out_channels каналов

        return dec3


def create_model(in_channels, out_channels):
    return UNet(in_channels, out_channels)
