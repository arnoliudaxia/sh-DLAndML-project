import os
import pickle
import numpy as np
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

root_dir='/home/arno/Projects/EEGDecodingTest/My/Data/LittlePrince'
for sub_dir in os.listdir(root_dir):
    sub_dir_path = os.path.join(root_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(sub_dir_path, file_name)
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    cut_eeg_data, texts, text_embeddings = loaded_data
                    RunData=[]
                    for eeg_data in cut_eeg_data:
                        # 每一段文本
                        freq_data = []
                        for channel in eeg_data:
                            fft_values = rfft(channel)
                            freqs = rfftfreq(len(channel), d=1/256)
                            # Extract the target frequency range
                            target_indices = (freqs >= 1) & (freqs <= 100)
                            freq_data.append(np.abs(fft_values[target_indices]))
                        
                        freq_data=np.array(freq_data)
                        RunData.append(freq_data)

                with open(file_path+'_dft.pkl', 'wb') as file:
                    print(file_path+'_dft.pkl')
                    pickle.dump([RunData, texts, text_embeddings], file)