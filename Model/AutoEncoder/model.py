# Step 2: Define the Autoencoder
import torch.nn as nn

class EEGAutoencoder(nn.Module):
    def __init__(self, input_channels=128, latent_dim=64, input_convTo=64, is_return_latent=False):
        super(EEGAutoencoder, self).__init__()
        self.is_return_latent=is_return_latent

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(16 * 32, latent_dim)  # Adjust 32 based on input
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 16 * 32)  # Match latent dimension
        self.unflatten = nn.Unflatten(1, (16, 32))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        batch_size, channels, seq_length = x.size()
        
        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc_latent(x)
        
        # Decoder
        x = self.fc_decode(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)
        if self.is_return_latent:
            return reconstructed[:, :, :seq_length], latent  # Crop to match input length
        else:
            return reconstructed[:, :, :seq_length]  # Crop to match input length
        
