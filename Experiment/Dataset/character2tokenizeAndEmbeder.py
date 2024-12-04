import os
import pickle
import numpy as np
import torch
from Model.AutoEncoder.model import EEGAutoencoder
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda")
model_name ='/home/arno/Projects/EEGDecodingTest/My/LLM/Qwen2.5-7B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'
with torch.no_grad():
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                # if file_name.endswith('.pkl') and 'latent' in file_name and '_tokenid' not in file_name:
                if file_name.endswith('.pkl') and '_tokenEmbedding' not in file_name:
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)
                    
                    tokenidsBatch=tokenizer( [d[1] for d in loaded_data], return_tensors="pt", padding=True)['input_ids']
                    tokenEmbeddingBatch=model.get_input_embeddings()(tokenidsBatch.to(device)).cpu().to(torch.float32).numpy().squeeze()
                    # breakpoint()
                    processed_data = [[eeg, sen, latent, tokenid.cpu().numpy(), embed] 
                                      for (eeg, sen, latent, _), tokenid,embed in zip(loaded_data, tokenidsBatch, tokenEmbeddingBatch )]  # 用于存储处理后的数据

                    # 分离文件名和扩展名
                    base, ext = os.path.splitext(file_path)
                    # file_path=base+"_tokenid.pkl"
                    file_path=base+"_tokenEmbedding.pkl"
                    
                    with open(file_path, 'wb') as output_file:
                        pickle.dump(processed_data, output_file)
                        print(f"Processed file saved: {file_path}")

