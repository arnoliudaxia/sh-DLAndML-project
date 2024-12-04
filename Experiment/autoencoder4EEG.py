import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import wandb
import argparse
from sklearn.model_selection import train_test_split
from My.Experiment.autoencoder4EEG import EEGAutoencoder
from Model.dataloader import EEGDataset

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchSize', type=int, default=1024)
parser.add_argument('--UseWandb', action='store_true') # 默认False
args = parser.parse_args()
print(args)



if args.UseWandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="EEG-project",
        name="CharacterSplitTest",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "pipeline-dataprocess": "ChanelWiseWhiteAndCharacterSplit",
        "pipeline-EEGEncoder": 'autoEncoder',
        'batch-size':args.batchSize,
        }
    )   

# Load the dataset
root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'  # Root directory containing the data folders (sub04, sub05, etc.)
dataset = EEGDataset(root_dir=root_dir)
# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=42
)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False)

dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
device = torch.device("cuda")
model = EEGAutoencoder().to(device)
if args.UseWandb:
    wandb.watch(model)
# Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Step 3: Training Loop with Validation
num_epochs = args.epochs
best_val_loss = float('inf')  # Track the best validation loss
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Log training and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if args.UseWandb:
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_eeg_autoencoder_{epoch}.pth")
        print("Saved best model with validation loss: {:.4f}".format(best_val_loss))

# Save the model and optimizer states
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': epoch,
#     'loss': epoch_loss
# }, "eeg_autoencoder_checkpoint.pth")
# print("Checkpoint saved.")
