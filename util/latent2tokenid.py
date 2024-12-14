import json
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from latent2embed import LatentToEmbedModel
from tqdm import tqdm

root_dir = '/media/dell/DATA/BoBo/sh-DLAndML-project-master/Data/qwen-characterSplit'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_data(root_dir):
    latents = []
    tokenids = []

    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.pkl') and 'latent' in file_name and '_tokenid' in file_name:
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)

                    for data in loaded_data:
                        # 确保每个数据点包含需要的部分
                        if len(data) >= 4:
                            _, _, latent, tokenid = data[:4]
                            latents.append(latent.squeeze())  # 移除多余的维度
                            
                            tokenid = tokenid[0] # TODO:
                            tokenids.append(tokenid)

    latents = torch.tensor(np.array(latents), dtype=torch.float32)
    tokenids = torch.tensor(np.array(tokenids), dtype=torch.long)  # tokenid为整数，使用long类型
    
    return latents, tokenids

def load_data_for_sub(root_dir, test_sub):
    train_latents = []
    train_tokenids = []
    test_latents = []
    test_tokenids = []
    
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
                        if len(data) >= 4:
                            _, _, latent, tokenid = data[:4]
                            tokenid = tokenid[0] # TODO:
                          
                            if is_test:
                                test_latents.append(latent.squeeze())
                                test_tokenids.append(tokenid)
                            else:
                                train_latents.append(latent.squeeze())
                                train_tokenids.append(tokenid)
    return train_latents, train_tokenids, test_latents, test_tokenids


class LatentTokenidDataset(Dataset):
    def __init__(self, latents, tokenids):
        self.latents = latents
        self.tokenids = tokenids

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.tokenids[idx]

# # 定义神经网络模型
# class LatentToTokenidModel(nn.Module):
#     def __init__(self, input_dim=64, hidden_dims=[512, 1024, 2048, 3584], output_dim=1456):
#         # 假设tokenid的种类数为10000，可以根据实际情况调整
#         super(LatentToTokenidModel, self).__init__()
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.3))  # 防止过拟合
#             prev_dim = hidden_dim
#         layers.append(nn.Linear(prev_dim, output_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

class LatentToTokenidModel(nn.Module):
    def __init__(self, embed_model: LatentToEmbedModel, num_classes=1456):
        super(LatentToTokenidModel, self).__init__()
        self.embed_model = embed_model
        # 冻结基础模型的参数
        for param in self.embed_model.parameters():
            param.requires_grad = False
        # 添加一个用于分类的线性层
        self.classifier = nn.Linear(self.embed_model.model[-1].out_features, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            embedding = self.embed_model(x)
        logits = self.classifier(embedding)
        return logits

def train(model, train_dataloader, criterion, optimizer, test_dataloader=None, model_save_dir='models', epochs=1000, save_every=10, model_name='latent_to_tokenid_model'):
    model.train()
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 使用 tqdm 显示进度
        with tqdm(train_dataloader, desc=f"Epoch [{epoch}/{epochs}]", unit="batch") as tepoch:
            for batch_latent, batch_tokenid in tepoch:
                batch_latent = batch_latent.to(device)
                batch_tokenid = batch_tokenid.to(device)

                # 前向传播
                outputs = model(batch_latent)
                loss = criterion(outputs, batch_tokenid)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累积损失
                epoch_loss += loss.item() * batch_latent.size(0)

                # 计算预测
                _, predicted = torch.max(outputs, 1)

                # 更新正确和总计数
                correct_predictions += (predicted == batch_tokenid).sum().item()
                total_predictions += batch_tokenid.size(0)

                # 计算当前批次的准确率
                batch_accuracy = (predicted == batch_tokenid).sum().item() / batch_tokenid.size(0) * 100

                # 更新 tqdm 显示的损失和准确率
                tepoch.set_postfix(loss=loss.item(), accuracy=f"{batch_accuracy:.2f}%")

        # 计算平均损失和整体准确率
        average_loss = epoch_loss / len(train_dataloader.dataset)
        epoch_accuracy = correct_predictions / total_predictions * 100

        # 打印轮次指标
        print(f'Epoch [{epoch}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # 每 save_every 轮次进行评估和保存模型
        if epoch % save_every == 0:
            if test_dataloader is not None:
                test_loss = evaluate(model, test_dataloader, criterion)
                print(f'Validation Loss after Epoch {epoch}: {test_loss:.4f}')

            # 保存模型
            model_save_path = os.path.join(model_save_dir, f'latent_to_tokenid_model_epoch_{epoch}_{model_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to '{model_save_path}'")

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_latent, batch_tokenid in dataloader:
            batch_latent = batch_latent.to(device)
            batch_tokenid = batch_tokenid.to(device)
            outputs = model(batch_latent)
            loss = criterion(outputs, batch_tokenid)
            total_loss += loss.item() * batch_latent.size(0)

            # 计算预测准确率
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_tokenid).sum().item()
            total_predictions += batch_tokenid.size(0)
    average_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions * 100
    print(f'Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    model.train()
    return average_loss

if __name__ == '__main__':
    print("Loading data...")
    root_dir = '/media/dell/DATA/BoBo/sh-DLAndML-project-master/Data/qwen-characterSplit'  # 请替换为您的数据路径
    latents, tokenids = load_data(root_dir)
    print(f'Loaded {len(latents)} samples.')

    unique_tokenids = torch.unique(tokenids)
    num_token_ids = unique_tokenids.size(0)
    print(f'Number of unique token ids: {num_token_ids}')

    tokenid_to_index = {tokenid.item(): idx for idx, tokenid in enumerate(unique_tokenids)}
    index_to_tokenid = {idx: tokenid.item() for idx, tokenid in enumerate(unique_tokenids)}

    # 保存映射（可选）
    with open('tokenid_to_index.json', 'w') as f:
        json.dump(tokenid_to_index, f)
    with open('index_to_tokenid.json', 'w') as f:
        json.dump(index_to_tokenid, f)

    sub_dirs = os.listdir(root_dir)  # 获取所有子目录名称
    batch_size = 2048
    epochs = 100
    save_every = 10  # 每10轮保存和评估一次
    results = {}  # 用于记录测试结果

    for test_sub in sub_dirs:
        print(f"\nProcessing with '{test_sub}' as the test set...")

        # 加载数据，将一个子目录作为测试集
        train_latents, train_tokenids, test_latents, test_tokenids = load_data_for_sub(root_dir, test_sub)

        # 打印数据集大小
        print(f"Training data: {len(train_latents)} samples")
        print(f"Testing data: {len(test_latents)} samples")

        # 映射 token IDs 到索引
        train_mapped_tokenids = torch.tensor([tokenid_to_index[tid.item()] for tid in train_tokenids], dtype=torch.long)
        test_mapped_tokenids = torch.tensor([tokenid_to_index[tid.item()] for tid in test_tokenids], dtype=torch.long)

        # 创建 Dataset 和 DataLoader
        train_dataset = LatentTokenidDataset(train_latents, train_mapped_tokenids)
        test_dataset = LatentTokenidDataset(test_latents, test_mapped_tokenids)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        latent2embed_model = LatentToEmbedModel().to(device)
        model_path = "/media/dell/DATA/BoBo/sh-DLAndML-project-master/latent_to_embed_model.pth"
        
        latent2embed_model.load_state_dict(torch.load(model_path, map_location=device))
        
        model = LatentToTokenidModel(embed_model=latent2embed_model, num_classes=num_token_ids).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 训练模型，传入测试集进行评估和模型保存
        print(f"Training the model with '{test_sub}' as the test set...")
        train(model, train_dataloader, criterion, optimizer, test_dataloader=test_dataloader, epochs=epochs, save_every=save_every, model_name=test_sub)

        # 在最终训练后对测试集进行一次评估
        final_test_loss = evaluate(model, test_dataloader, criterion)
        print(f"Final Test Loss for '{test_sub}': {final_test_loss:.4f}")

        # 保存最终模型（包括测试集名称）
        final_model_save_path = f'latent_to_tokenid_model_final_{test_sub}.pth'
        torch.save(model.state_dict(), final_model_save_path)
        print(f"Final model saved as '{final_model_save_path}'")
    
    
    
    
    
    
    
    
    

    # # 划分数据集为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(latents, tokenids, test_size=0.1, random_state=42)

    # # 创建对应的 Dataset 和 DataLoader
    # train_dataset = LatentTokenidDataset(X_train, y_train)
    # test_dataset = LatentTokenidDataset(X_test, y_test)

    # batch_size = 4096
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 初始化模型、损失函数和优化器
    # num_token_ids = len(torch.unique(tokenids))  # 根据数据确定tokenid类别数
    
    # print(f'Number of unique token ids: {num_token_ids}')
    
    # max_tokenid = torch.max(tokenids)
    
    # model = LatentToTokenidModel(output_dim=max_tokenid + 1).to(device)
    # criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # # 仅在训练集上训练
    # train(model, train_dataloader, criterion, optimizer, epochs=2000)

    # # 在测试集上评估
    # test_loss = evaluate(model, test_dataloader, criterion)
    # print(f'Test Loss: {test_loss}')

    # # 保存模型
    # torch.save(model.state_dict(), 'latent_to_tokenid_model.pth')
    # print("Model saved as 'latent_to_tokenid_model.pth'")