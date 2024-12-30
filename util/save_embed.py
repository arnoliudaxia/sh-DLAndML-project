import argparse
import numpy as np
import torch
import pickle
import os
from My.Model.latent2Embedding.LatentToEmbedModel import LatentToEmbedModelMLP, LatentToEmbedModelLinear
from My.Experiment.latent2Embedding.latent2embed import load_data_for_subs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = "My/Data/qwen-characterSplit"

# 数据加载目录和模型目录
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--mask', type=int, nargs='+', help="Mask掉的subject（不用做训练）")
args = parser.parse_args()
print(args)
model_path = args.model_path
output_dir = args.output_path
# 若输出目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载数据
_, _, latents, embeds = load_data_for_subs(root_dir, [f"sub{mask:02}" for mask in args.mask])
latents=np.array(latents)
latents_tensor = torch.tensor(latents, dtype=torch.float32).to(device)


print(f"Loading model: {model_path}")

# model = LatentToEmbedModelLinear(input_dim=2048).to(device)
model = LatentToEmbedModelMLP(input_dim=2048, hidden_dims=[2048,2048]).to(device) # ! Model选择

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 使用 no_grad() 推断
with torch.no_grad():
    preds = model(latents_tensor)
    preds = preds.cpu().numpy()

data_to_save = {
    'latents': latents,
    'true_embeds': embeds,
    'predicted_embeds': preds
}
embeds = torch.tensor(np.array(embeds), dtype=torch.float32)
preds = torch.tensor(np.array(preds), dtype=torch.float32)

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

# Now mean_squared_diff will have the shape [44648], representing the mean of each row (along the 3584 dimension)
euclidean_distances=[euclidean_distance(e,p) for e,p in zip(embeds,preds)]
print(np.array(euclidean_distances).mean())
# exit()
# 根据模型文件名生成对应的输出pkl文件名
output_filename =  'latentEmbedding_predictions.pkl'
output_path = os.path.join(output_dir, output_filename)

with open(output_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Predictions for {model_path} saved to {output_path}")
