import json
import os
import pickle
import torch
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

root_dir = '/media/dell/DATA/BoBo/sh-DLAndML-project-master/Data/qwen-characterSplit'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_data(root_dir):
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


def load_data_for_sub(root_dir, test_sub):
    train_latents = []
    train_embeds = []
    test_latents = []
    test_embeds = []
    
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            is_test = (sub_dir == test_sub)  # 当前目录是否为测试集
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
                                if is_test:
                                    test_latents.append(latent.squeeze())
                                    test_embeds.append(embed)
                                else:
                                    train_latents.append(latent.squeeze())
                                    train_embeds.append(embed)
    return train_latents, train_embeds, test_latents, test_embeds


def load_data_for_subs(root_dir, test_subs):
    """
    加载数据，将指定的多个子目录作为测试集，其余的作为训练集。

    参数：
    - root_dir (str): 根目录路径，包含所有子目录。
    - test_subs (list or tuple): 作为测试集的子目录名称列表。

    返回：
    - train_latents (list): 训练集的latent数据。
    - train_embeds (list): 训练集的embedding数据。
    - test_latents (list): 测试集的latent数据。
    - test_embeds (list): 测试集的embedding数据。
    """
    train_latents = []
    train_embeds = []
    test_latents = []
    test_embeds = []
    
    # 遍历根目录下的所有子目录
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            # 判断当前子目录是否在测试集列表中
            is_test = sub_dir in test_subs
            for file_name in os.listdir(sub_dir_path):
                # 过滤出符合条件的.pkl文件
                if (file_name.endswith('.pkl') and 'latent' in file_name 
                    and '_tokenid' in file_name and '_tokenEmbedding' in file_name):
                    file_path = os.path.join(sub_dir_path, file_name)
                    try:
                        with open(file_path, 'rb') as file:
                            loaded_data = pickle.load(file)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue  # 跳过无法加载的文件
                    
                    for data in loaded_data:
                        if len(data) >= 5:
                            _, _, latent, tokenid, embed = data
                            if embed.shape[0] == 3584:
                                if is_test:
                                    test_latents.append(latent.squeeze())
                                    test_embeds.append(embed)
                                else:
                                    train_latents.append(latent.squeeze())
                                    train_embeds.append(embed)
    
    return train_latents, train_embeds, test_latents, test_embeds

class LatentEmbedDataset(Dataset):
    def __init__(self, latents, embeds):
        self.latents = latents
        self.embeds = embeds
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.embeds[idx]

# 定义神经网络模型
class LatentToEmbedModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[512, 1024, 2048], output_dim=3584):
        super(LatentToEmbedModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # 防止过拟合
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
# class LatentToEmbedLSTM(nn.Module):
#     def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, output_dim=3584):
#         super(LatentToEmbedLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
    # def forward(self, x):
    #     # 打印输入形状以进行调试
    #     # print(f"Input shape before adjustment: {x.shape}")
        
    #     # 检查输入维度并调整
    #     if x.dim() == 2:
    #         # 添加序列长度维度，假设序列长度为1
    #         x = x.unsqueeze(1)  # 形状变为 (batch_size, 1, input_dim)
    #         # print(f"Input shape after unsqueeze: {x.shape}")
    #     elif x.dim() == 3:
    #         # print(f"Input shape is already 3D: {x.shape}")
    #         pass
    #     else:
    #         raise ValueError(f"Expected input to be 2D or 3D, but got {x.dim()}D")
        
    #     # 通过 LSTM 层
    #     lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out 形状为 (batch_size, seq_length, hidden_dim)
    #     # print(f"LSTM output shape: {lstm_out.shape}")
        
    #     # 获取最后一个时间步的输出
    #     last_out = lstm_out[:, -1, :]  # 形状为 (batch_size, hidden_dim)
    #     # print(f"Last output shape: {last_out.shape}")
        
    #     # 通过全连接层
    #     out = self.fc(last_out)  # 形状为 (batch_size, output_dim)
    #     # print(f"Output shape: {out.shape}")
        
    #     return out

def train(model, dataloader, criterion, optimizer, epochs=1000):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_latent, batch_embed in dataloader:
            batch_latent = batch_latent.to(device)
            batch_embed = batch_embed.to(device)
            
            # 前向传播
            outputs = model(batch_latent)
            loss = criterion(outputs, batch_embed)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_latent.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss}')

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_latent, batch_embed in dataloader:
            batch_latent = batch_latent.to(device)
            batch_embed = batch_embed.to(device)
            outputs = model(batch_latent)
            loss = criterion(outputs, batch_embed)
            total_loss += loss.item() * batch_latent.size(0)
    total_loss /= len(dataloader.dataset)
    return total_loss

if __name__ == '__main__':
    
    # print("Loading data...")
    # latents, embeds = load_data(root_dir)
    # print(f'Loaded {len(latents)} samples.')

    # # 划分数据集为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(latents, embeds, test_size=0.1, random_state=42)

    # # 创建对应的 Dataset 和 DataLoader
    # train_dataset = LatentEmbedDataset(X_train, y_train)
    # test_dataset = LatentEmbedDataset(X_test, y_test)

    # batch_size = 2048
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 初始化模型、损失函数和优化器
    # model = LatentToEmbedLSTM().to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # # 仅在训练集上训练
    # train(model, train_dataloader, criterion, optimizer, epochs=500)

    # # 在测试集上评估
    # test_loss = evaluate(model, test_dataloader, criterion)
    # print(f'Test Loss: {test_loss}')

    # # 保存模型
    # torch.save(model.state_dict(), 'latent_to_embed_model_lstm.pth')
    # print("Model saved as 'latent_to_embed_model_lstm.pth'")


    print("Loading data...")

    sub_dirs = os.listdir(root_dir)  # 获取所有子目录名称
    batch_size = 2048
    epochs = 500
    results = {}  # 用于记录测试结果

    # 生成所有可能的3个子目录组合
    test_combinations = list(itertools.combinations(sub_dirs, 3))
    total_combinations = len(test_combinations)
    print(f"Total test combinations: {total_combinations}")

    for idx, test_subs in enumerate(test_combinations, 1):
        test_subs_str = "_".join(test_subs)
        print(f"\nProcessing combination {idx}/{total_combinations} with '{test_subs_str}' as the test set...")

        # 加载数据，将三个子目录作为测试集
        train_latents, train_embeds, test_latents, test_embeds = load_data_for_subs(root_dir, test_subs)

        # 打印数据集大小
        print(f"Training data: {len(train_latents)} samples")
        print(f"Testing data: {len(test_latents)} samples")

        # 创建对应的 Dataset 和 DataLoader
        train_dataset = LatentEmbedDataset(train_latents, train_embeds)
        test_dataset = LatentEmbedDataset(test_latents, test_embeds)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = LatentToEmbedModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 仅在训练集上训练
        print(f"Training the model with '{test_subs_str}' as the test set...")
        train(model, train_dataloader, criterion, optimizer, epochs=epochs)

        # 在测试集上评估
        test_loss = evaluate(model, test_dataloader, criterion)
        print(f"Test Loss for '{test_subs_str}': {test_loss}")

        # 保存模型（带上测试集名称）
        model_save_path = f'latent_to_embed_model_{test_subs_str}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved as '{model_save_path}'")

        # 将测试结果记录到字典中
        results[test_subs_str] = test_loss

    # 将结果保存到 JSON 文件
    results_file = 'test_results.json'
    with open(results_file, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"All results saved to '{results_file}'")