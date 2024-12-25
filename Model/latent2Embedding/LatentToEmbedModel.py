import torch.nn as nn
import torch


class LatentToEmbedModelMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[512, 1024, 2048], output_dim=3584):
        super(LatentToEmbedModelMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 使用LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 防止过拟合
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class LatentToEmbedModelLinear(nn.Module):
    def __init__(self, input_dim=2048, output_dim=3584):
        super(LatentToEmbedModelLinear, self).__init__()
        # 线性投影：从输入维度映射到输出维度
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)
    
    


class WeightedMSECosLoss(nn.Module):
    def __init__(self, cos_weight=0.8):
        """
        初始化损失函数。
        参数:
        - cos_weight: Cosine loss 的权重 (0-1)。
        """
        super(WeightedMSECosLoss, self).__init__()
        self.cos_weight = cos_weight
        self.mse_weight = 1.0 - cos_weight
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    def forward(self, output, target):
        """
        计算加权损失。
        参数:
        - output: 模型的预测输出 (batch_size, embedding_dim)。
        - target: 目标向量 (batch_size, embedding_dim)。
        返回:
        - 综合损失值。
        """
        # 计算 MSELoss
        mse_loss = self.mse_loss(output, target)
        
        # 计算 Cosine Loss (1 - cosine similarity)
        cos_sim = self.cosine_similarity(output, target)
        cos_loss = (1 - cos_sim).mean()

        # 加权平均损失
        total_loss = self.mse_weight * mse_loss + self.cos_weight * cos_loss
        return total_loss