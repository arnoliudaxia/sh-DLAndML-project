import argparse
import json
import os
import pickle
import torch
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from My.Model.latent2Embedding.LatentToEmbedModel import LatentToEmbedModelLinear, WeightedMSECosLoss, LatentToEmbedModelMLP


root_dir = "My/Data/qwen-characterSplit"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
            print(f"{sub_dir} is Test? {is_test}")
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
                            if embed.shape[0] != 3584:
                                embed=embed[0,:]
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


def euclidean_distance(tensor1, tensor2):
    """
    计算两个张量之间的欧几里得距离。
    
    参数:
    - tensor1 (torch.Tensor): 第一个张量，形状为 (..., D)，其中 D 是特征维度。
    - tensor2 (torch.Tensor): 第二个张量，形状为 (..., D)。
    
    返回:
    - distance (torch.Tensor): 欧几里得距离，形状为 (...,)。
    """
    # 确保两个张量在相同设备上
    # tensor1 = tensor1.to(tensor2.device)
    if type(tensor1)!=torch.Tensor:
        tensor1=torch.tensor(tensor1)
        tensor2=torch.tensor(tensor2)
    
    # 计算平方差
    diff = tensor1 - tensor2
    squared_diff = diff ** 2
    
    # 求和并取平方根
    distance =torch.sqrt(torch.sum(squared_diff, dim=-1))
    return distance


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

def evaluateMetric(model, dataloader, criterion):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_latent, batch_embed in dataloader:
            batch_latent = batch_latent.to(device)
            batch_embed = batch_embed.to(device)
            outputs = model(batch_latent)
            total_loss.extend([euclidean_distance(outputs[i], batch_embed[i]) for i in range(len(batch_embed))])

    return torch.tensor(total_loss).mean().item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--SaveModelPath', type=str, default=None) 
    parser.add_argument('--UseWandb', action='store_true') # 默认False
    parser.add_argument('--batchSize', type=int, default=2048)
    parser.add_argument('--mask', type=int, nargs='+', help="Mask掉的subject（不用做训练）")
    
    args = parser.parse_args()
    print(args)
    batch_size = args.batchSize
    
    if args.SaveModelPath is not None:
        os.makedirs(args.SaveModelPath, exist_ok=True)
    print("Loading data...")
    if args.UseWandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="EEG-project",
            name=f"mask{'_'.join(map(str, args.mask))}-hidden2048",
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "pipeline-dataprocess": "ChanelWiseWhiteAndCharacterSplit",
            "pipeline-EEGEncoder": 'latent2Embedding',
            'batch-size':args.batchSize,
            }
        )   
    # sub_dirs = os.listdir(root_dir)  # 获取所有子目录名称
    sub_dirs = [[f"sub{mask:02}" for mask in args.mask]] 
    epochs = 100
    results = {}  # 用于记录测试结果
    

    # 生成所有可能的3个子目录组合
    # test_combinations = list(itertools.combinations(sub_dirs, 3))
    test_combinations=sub_dirs
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
        # Set up DataLoader for training and validation splits
        val_split = 0.1  # 10% of the training data will be used for validation
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        best_val_loss = float('inf')  # Track the best validation loss
        # model = LatentToEmbedModelLinear(input_dim=2048).to(device) # ! Model选择
        model = LatentToEmbedModelMLP(input_dim=2048, hidden_dims=[2048,2048]).to(device) # ! Model选择
        
        if args.UseWandb:
            wandb.watch(model)
        criterion = nn.MSELoss() # ! 改一下LOSS
        # criterion = WeightedMSECosLoss(cos_weight=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train the model with a validation set
        print(f"Training the model with '{test_subs_str}' as the test set...")
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            val_loss = 0.0
            # Training loop
            for batch_latent, batch_embed in train_dataloader:
                batch_latent = batch_latent.to(device)
                batch_embed = batch_embed.to(device)

                # Forward pass
                outputs = model(batch_latent)
                loss = criterion(outputs, batch_embed)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_latent.size(0)

            epoch_loss /= len(train_dataloader.dataset)

            # Validation loop
            model.eval()
            with torch.no_grad():
                for batch_latent, batch_embed in val_dataloader:
                    batch_latent = batch_latent.to(device)
                    batch_embed = batch_embed.to(device)

                    outputs = model(batch_latent)
                    loss = criterion(outputs, batch_embed)

                    val_loss += loss.item() * batch_latent.size(0)

            val_loss /= len(val_dataloader.dataset)
            
            
            # Evaluate on the test set
            test_euclidean_distance = evaluateMetric(model, test_dataloader, criterion)

            # print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss}, Validation Loss: {val_loss}, Test Loss for {test_subs_str}: {test_loss}')
            print(f'Epoch [{epoch}/{epochs}], '
                        f'Train Loss: {epoch_loss:.3e}, '
                        f'Validation Loss: {val_loss:.3e}, '
                        f'Test euclidean mean distance {test_subs_str}: {test_euclidean_distance:.3e}')
           
            # Log losses to wandb if using it
            if args.UseWandb:
                wandb.log({'train_loss': epoch_loss, 'val_loss': val_loss, 'test_euclidean_distance': test_euclidean_distance})
            # Save the best model
            if args.SaveModelPath is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = f'{args.SaveModelPath}/latent_to_embed_model_{test_subs_str}.pth'
                torch.save(model.state_dict(), model_save_path)
                print("Saved best model with validation loss: {:.4f}".format(best_val_loss))
                

