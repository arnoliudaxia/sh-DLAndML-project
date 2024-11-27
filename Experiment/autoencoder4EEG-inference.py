import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from util.dataprocess import whiten_data

# Step 1: Define the EEG Dataset
class EEGDataset(Dataset):
    def __init__(self, root_dir, target_length=256):
        self.data_files = []
        self.target_length = target_length
        self.load_data(root_dir)

    def load_data(self, root_dir):
        for sub_dir in os.listdir(root_dir):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                for file_name in os.listdir(sub_dir_path):
                    if file_name.endswith('.pkl'):
                        file_path = os.path.join(sub_dir_path, file_name)
                        with open(file_path, 'rb') as file:
                            loaded_data = pickle.load(file)
                            cut_eeg_data, _, _ = loaded_data
                            self.data_files.extend(cut_eeg_data)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        eeg_data = self.data_files[idx]
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)  # Shape: (128, variable_length)
        
        # Apply Adaptive Pooling to ensure fixed length
        if eeg_data.shape[1] != self.target_length:
            eeg_data = nn.functional.adaptive_avg_pool1d(eeg_data.unsqueeze(0), self.target_length).squeeze(0)
        
        return eeg_data

# Step 2: Define the Autoencoder
class EEGAutoencoder(nn.Module):
    def __init__(self, input_channels=128, latent_dim=64):
        super(EEGAutoencoder, self).__init__()
        
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
        return latent, reconstructed[:, :, :seq_length]  # Crop to match input length


root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/LittlePrince'  # Root directory containing the data folders (sub04, sub05, etc.)
dataset = EEGDataset(root_dir=root_dir)
channel_mean=np.load('channel_mean.npy')
channel_std=np.load('channel_std.npy')

def whitening_collate_fn(batch): # 定义白化后的 DataLoader
    batch = torch.stack(batch)  # Shape: (batch_size, channels, seq_length)
    whitened_batch = []
    for data in batch:
        whitened_data = whiten_data(data, channel_mean, channel_std)
        whitened_batch.append(whitened_data)
    return torch.stack(whitened_batch)

dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=whitening_collate_fn)
device = torch.device("cuda")
model = EEGAutoencoder().to(device)

# Load the checkpoint
checkpoint = torch.load("eeg_autoencoder_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
latent_outputs = []
decoder_outputs = []
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        latent, reconstructed = model(batch)
        breakpoint()
        latent_outputs.append(latent.cpu())
        decoder_outputs.append(reconstructed.cpu())

latent_outputs = torch.cat(latent_outputs, dim=0)
decoder_outputs = torch.cat(decoder_outputs, dim=0)

print("Latent representations shape:", latent_outputs.shape)
print("Decoder outputs shape:", decoder_outputs.shape)
