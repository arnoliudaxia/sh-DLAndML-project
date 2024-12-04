import os
import pickle
import re
import sys
import numpy as np
sys.path.append('/home/arno/Projects/EEGDecodingTest/My')
from util.dataprocess import whiten_data

channel_mean, channel_std=np.load('channel_mean.npy'),np.load('channel_std.npy')

root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'

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
                        eeg=whiten_data(eeg, channel_mean, channel_std)
                        processed_data.append([eeg,sen])
                        
                with open(file_path, 'wb') as output_file:
                    pickle.dump(processed_data, output_file)
                    print(f"Processed file saved: {file_path}")
