import os
import pickle
import numpy as np
import torch
from Model.AutoEncoder.model import EEGAutoencoder
from torch import nn



device = torch.device("cuda")
model = EEGAutoencoder(is_return_latent=True).to(device)

# Load the checkpoint
checkpoint = torch.load("Model/AutoEncoder/eeg_autoencoder_checkpoint.pth", weights_only=True)
model.load_state_dict(checkpoint)

model.eval()

root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'
with torch.no_grad():

    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)
                    processed_data = []  # 用于存储处理后的数据
                    for eeg, sen in loaded_data:
                        eeg = np.expand_dims(eeg, axis=0)
                        eeg=torch.tensor(eeg, dtype=torch.float32).to(device=device)
                        eeg = nn.functional.adaptive_avg_pool1d(eeg, 256)
                        
                        # breakpoint()
                        _, latent = model(eeg)
                        processed_data.append([eeg,sen, latent])
                    
                    # 分离文件名和扩展名
                    base, ext = os.path.splitext(file_path)
                    file_path=base+"_with_latent.pkl"
                    
                    with open(file_path, 'wb') as output_file:
                        pickle.dump(processed_data, output_file)
                        print(f"Processed file saved: {file_path}")
                    
                        # print("Latent representations shape:", latent.shape)
                        # print("eeg shape:", eeg.shape)

