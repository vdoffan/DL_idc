import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=2),
        nn.ReLU(inplace=True)
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1),
        nn.ReLU(inplace=True)
    )

    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()


        # добавьте несколько слоев encoder block
        # это блоки-составляющие энкодер-части сети
        self.encoder = nn.Sequential(
            encoder_block(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            encoder_block(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            encoder_block(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        )

        # добавьте несколько слоев decoder block
        # это блоки-составляющие декодер-части сети
        self.decoder = nn.Sequential(
            decoder_block(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            decoder_block(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # Последний слой восстанавливает до 3 каналов для исходного формата (RGB)
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.Sigmoid()  # sigmoid для нормализации выхода в [0,1], опционально
        )

    def forward(self, x):

        # downsampling 
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction



def create_model():
    return Autoencoder()