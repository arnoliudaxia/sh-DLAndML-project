import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn as nn

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
                            for e in loaded_data:
                                # breakpoint()
                                self.data_files.append(e[0][0])


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        eeg_data = self.data_files[idx]
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)  # Shape: (128, variable_length)
        
        # Apply Adaptive Pooling to ensure fixed length
        if eeg_data.shape[1] != self.target_length:
            eeg_data = nn.functional.adaptive_avg_pool1d(eeg_data.unsqueeze(0), self.target_length).squeeze(0)
        
        return eeg_data


if __name__ == '__main__':
    # Example usage
    root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'  # Root directory containing the data folders (sub04, sub05, etc.)
    dataset = EEGDataset(root_dir=root_dir)
    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")