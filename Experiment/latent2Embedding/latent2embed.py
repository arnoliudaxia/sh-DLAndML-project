import argparse
import json
import os
import pickle
import torch.nn.functional as F
import torch
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from My.Model.latent2Embedding.LatentToEmbedModel import LatentToEmbedModelLinear, WeightedMSECosLoss, LatentToEmbedModelMLP
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics.pairwise import cosine_similarity

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
    characters=[]
    
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
                            _, ch, latent, tokenid, embed = data
                            if embed.shape[0] != 3584:
                                embed=embed[0,:]
                            if is_test:
                                test_latents.append(latent.squeeze())
                                test_embeds.append(embed)
                            else:
                                train_latents.append(latent.squeeze())
                                train_embeds.append(embed)
                                characters.append(ch)
    return train_latents, train_embeds, test_latents, test_embeds, characters

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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # 计算欧几里得距离
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # 计算Contrastive Loss
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def euclidean_distance_loss(predictions, targets):
    return torch.sum((predictions - targets) ** 2)
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

def evaluateMetric(model, dataloader):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_latent, batch_embed in dataloader:
            batch_latent = batch_latent.to(device)
            batch_embed = batch_embed.to(device)
            outputs = model(batch_latent)
            total_loss.extend([euclidean_distance(outputs[i], batch_embed[i]) for i in range(len(batch_embed))])

    return torch.tensor(total_loss).mean().item()

def create_labels(batch_embed):
    # 计算Cosine相似度矩阵
    # 使用归一化来计算Cosine相似度
    norm_batch_embed = F.normalize(batch_embed, p=2, dim=1)  # 归一化每个样本向量
    cosine_sim = torch.matmul(norm_batch_embed, norm_batch_embed.T)  # 计算相似度矩阵

    # 将相似度矩阵与一个阈值（例如0.99）比较，决定正负样本对
    labels = (cosine_sim > 0.99).float()  # 相似度大于阈值认为是正样本对

    # 将对角线上的值设置为0，因为它们是自己与自己的比较
    labels.fill_diagonal_(0)

    return labels

def train(args):
    sub_dirs = [[f"sub{mask:02}" for mask in args.mask]] 
    epochs = 200
    batch_size=args.batchSize
    if args.SaveModelPath is not None:
        os.makedirs(args.SaveModelPath, exist_ok=True)

    test_combinations=sub_dirs
    test_subs=test_combinations[0]
    test_subs_str = "_".join(test_subs)
    print(f"\nProcessing with '{test_subs_str}' as the test set...")

    # 加载数据，将三个子目录作为测试集
    train_latents, train_embeds, test_latents, test_embeds, chs = load_data_for_subs(root_dir, test_subs)
    # pickle.dump((train_latents, train_embeds, test_latents, test_embeds), open(f"MaskTrain-{test_subs_str}.pkl", "wb"))
    # train_latents, train_embeds, test_latents, test_embeds=pickle.load(open(f"MaskTrain-{test_subs_str}.pkl", "rb"))
        

    # 1. 创建标签映射字典
    unique_labels = list(set(chs))  # 获取唯一标签
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # 将标签映射到整数
    # 2. 根据 chs 列表生成整数标签
    integer_labels = [label_map[ch] for ch in chs]
    
    # 2. 使用RandomUnderSampler进行欠采样
    X = np.array(train_latents )
    y = np.array(integer_labels)

    # 2. 使用RandomUnderSampler进行欠采样
    undersampler = RandomUnderSampler(random_state=42)

    # 对数据进行欠采样
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # 3. 还原数据格式（train_latents 和 train_embeds）
    train_latents_resampled = X_resampled
    train_embeds_resampled = [train_embeds[i] for i in undersampler.sample_indices_]

    print(f"Original number of samples: {len(train_latents)}")
    print(f"Resampled number of samples: {len(train_latents_resampled)}")

    train_latents=train_latents_resampled
    train_embeds=train_embeds_resampled
    
    # 创建对应的 Dataset 和 DataLoader
    train_dataset = LatentEmbedDataset(train_latents, train_embeds)
    test_dataset = LatentEmbedDataset(test_latents, test_embeds)
    print(f"Training data: {len(train_latents)} samples")
    print(f"Testing data: {len(test_latents)} samples")
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
    model = LatentToEmbedModelMLP(input_dim=2048, hidden_dims=[args.fc_layer_size]*args.depthOfMLP).to(device) # ! Model选择
    
    if args.UseWandb:
        wandb.watch(model)
        
    if args.LossType=="MSE":
        criterion = nn.MSELoss()
    elif args.LossType=="ContrastiveLoss":
        print("Using ContrastiveLoss with margin=0.01")
        criterion = ContrastiveLoss(margin=0.01)
        # criterion = ContrastiveLoss(margin=0.1)
        # criterion = ContrastiveLoss(margin=0.8)
    

    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 设置学习率调度器，监控验证集损失，当验证集损失不再改善时将学习率减少为原来的 0.1 倍
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
            if args.LossType=="MSE":
                loss = criterion(outputs, batch_embed)
            elif args.LossType=="ContrastiveLoss":
                # 构造正负样本对： 这里我们假设每个batch中相同的`batch_embed`为正样本对，其他为负样本对
                # label为0表示负样本对，1表示正样本对
                labels = create_labels(batch_embed) 
                # 计算损失
                loss = criterion(outputs, batch_embed, labels)
            elif args.LossType=="Euclidean":
                loss = euclidean_distance_loss(outputs, batch_embed)
            
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
                # loss = criterion(outputs, batch_embed)
                labels = create_labels(batch_embed) 
                if args.LossType=="MSE":
                    loss = criterion(outputs, batch_embed)
                elif args.LossType=="ContrastiveLoss":
                    loss = criterion(outputs, batch_embed, labels)
                elif args.LossType=="Euclidean":
                    loss = euclidean_distance_loss(outputs, batch_embed)

                val_loss += loss.item() * batch_latent.size(0)

        val_loss /= len(val_dataloader.dataset)
        scheduler.step(val_loss)
        
        
        # Evaluate on the test set
        test_euclidean_distance = evaluateMetric(model, test_dataloader)

        # print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss}, Validation Loss: {val_loss}, Test Loss for {test_subs_str}: {test_loss}')
        print(f'Epoch [{epoch}/{epochs}], '
             f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}, "
                    f'Train Loss: {epoch_loss:.3e}, '
                    f'Validation Loss: {val_loss:.3e}, '
                    f'Test euclidean mean distance {test_subs_str}: {test_euclidean_distance:.3e}')
        
        # Log losses to wandb if using it
        if args.UseWandb:
            # wandb.log({'train_loss': epoch_loss, 'val_loss': val_loss})
            wandb.log({'train_loss': epoch_loss, 'val_loss': val_loss, 'test_euclidean_distance': test_euclidean_distance})
        # Save the best model
        if args.SaveModelPath is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f'{args.SaveModelPath}/latent_to_embed_model_{test_subs_str}.pth'
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model with validation loss: {:.4f}".format(best_val_loss))
            # if args.UseWandb:
            #      wandb.run.summary['test_euclidean_distance'] = test_euclidean_distance
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--SaveModelPath', type=str, default=None) 
    parser.add_argument('--LossType', type=str, default="MSE", help="loss的类型, [MSE, ContrastiveLoss]") 
    parser.add_argument('--UseWandb', action='store_true') # 默认False
    parser.add_argument('--batchSize', type=int, default=2048)
    parser.add_argument('--depthOfMLP', type=int, default=2, help="MLP的隐藏层深度")
    parser.add_argument('--fc_layer_size', type=int, default=2048, help="MLP的隐藏层宽度")
    parser.add_argument('--mask', type=int, nargs='+', help="Mask掉的subject（不用做训练）")
    parser.add_argument('--note', type=str, default=None) 
    
    args = parser.parse_args()
    print(args)
    batch_size = args.batchSize
    

    print("Loading data...")
    if args.UseWandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="EEG-project",
            name=f"mask{'_'.join(map(str, args.mask))}-{args.note}",
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "pipeline-dataprocess": "ChanelWiseWhiteAndCharacterSplit",
            "pipeline-EEGEncoder": 'latent2Embedding',
            'batch-size':args.batchSize,
            "loss":"ContrastiveLoss"
            },
            notes=args.note
        )   
    train(args)