import os
import pickle
from typing import Optional

 


def getAllData(root_dir = 'Data/qwen-characterSplit'):

    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.pkl') and 'latent' in file_name and '_tokenid'  in file_name and'_tokenEmbedding' in file_name:
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)
                    # 如果loaded_data里面每一个元素没有下面那么多成分，以每个的shape为准，shape总是对的。比如没有一个3584长度的东西说明embedding没在里面
                    for  eeg, character, latent, tokenid, embed in loaded_data:
                        # eeg.shape (1, 128, 256) 多通道EEG信号
                        # character -> str 一个中文汉字
                        # latent.shape (1, 64) EEG压缩后的向量
                        # tokenid array([24]) 汉字对应的token
                        # embed.shape (3584,)  词嵌入向量
                        breakpoint()
                        pass
                    
def getMaskData(root_dir = 'Data/qwen-characterSplit', subjectMask: Optional[list] = None):
    # 只获取mask掉的subject的数据，例如subjectMask=[8]则只返回subj8的数据
    latents = []
    embeds = []
    
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                if (file_name.endswith('.pkl') and 'latent' in file_name 
                    and '_tokenid' in file_name and '_tokenEmbedding' in file_name):
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)
                    
                    for data in loaded_data:

                        if len(data) >= 5:
                            _, _, latent, tokenid, embed = data

                            if embed.shape[0] == 3584:
                                latents.append(latent.squeeze())
                                embeds.append(embed)
    return latents, embeds
