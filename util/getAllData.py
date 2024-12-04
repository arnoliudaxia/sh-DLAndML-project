import os
import pickle


root_dir = 'Data/qwen-characterSplit'

for sub_dir in os.listdir(root_dir):
    sub_dir_path = os.path.join(root_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith('.pkl') and 'latent' in file_name and '_tokenid'  in file_name:
                file_path = os.path.join(sub_dir_path, file_name)
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                for  eeg, character, latent, tokenid in loaded_data:
                    # latent.shape (1, 64)
                    # tokenid array([24])
                    pass