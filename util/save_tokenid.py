import torch
import pickle
import os
import numpy as np
import json  # 新增：用于加载映射文件
from latent2tokenid import LatentToTokenidModel, load_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载映射文件
mapping_path = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/tokenid_to_index.json'  # 根据实际路径调整
index_to_tokenid_path = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/index_to_tokenid.json'  # 根据实际路径调整

# 加载 index_to_tokenid 映射
with open(index_to_tokenid_path, 'r') as f:
    index_to_tokenid = json.load(f)

# 如果 index_to_tokenid 的键是字符串，需要将其转换为整数
index_to_tokenid = {int(k): v for k, v in index_to_tokenid.items()}

print(f"Loaded index_to_tokenid mapping with {len(index_to_tokenid)} entries.")

# 加载数据
root_dir = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/Data/qwen-characterSplit'
latents, true_tokenids = load_data(root_dir)  # 确保变量名与返回值一致

# 初始化模型并加载权重
model = LatentToTokenidModel(output_dim=torch.unique(true_tokenids).size(0)).to(device)
model.load_state_dict(torch.load(
    r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/latent_to_tokenid_model_new2.pth', 
    map_location=device
))
model.eval()  # 设置为评估模式
print("Loaded trained model.")

# 确保 latents 是 (N, latent_dim) 的 numpy 数组
latents_tensor = torch.tensor(latents, dtype=torch.float32).to(device)  # 形状 (N, 64)

# 创建存储预测结果的文件夹
output_dir = '/media/dell/DATA/BoBo/sh-DLAndML-project-master/tokenid_predictions_new'
os.makedirs(output_dir, exist_ok=True)

# 分批处理
batch_size = 10240  # 根据显存大小调整批量大小
all_predictions = []

# 使用 no_grad() 关闭梯度计算，提高推断速度并节省内存
with torch.no_grad():
    for i in range(0, len(latents_tensor), batch_size):
        batch = latents_tensor[i:i + batch_size]
        logits = model(batch)  # 推理，输出形状 (batch_size, num_token_ids)
        
        print(logits.shape)
        print(logits)
        

        
        # 获取预测的索引
        predicted_indices = torch.argmax(logits, dim=1)  # 形状 (batch_size,)
        
        
        print(predicted_indices)

        # 将索引转换为原始的 token_id
        # 首先将 tensor 转为 numpy
        predicted_indices_np = predicted_indices.cpu().numpy()
        
        # 使用映射将索引转换为 token_id
        # 注意 index_to_tokenid 的键是整数索引
        predicted_tokenids = [index_to_tokenid[idx] for idx in predicted_indices_np]
        
        # 将预测的 token_id 转换为 numpy 数组
        predicted_tokenids_np = np.array(predicted_tokenids)
        
        all_predictions.append(predicted_tokenids_np)
        
        # 每处理一个批次就保存为一个 pkl 文件
        batch_number = i // batch_size + 1  # 计算批次编号
        data_to_save = {
            'latents': latents[i:i + batch_size],
            'true_tokenid': true_tokenids[i:i + batch_size],
            'predicted_tokenid': predicted_tokenids_np
        }
        # 生成保存路径
        output_path = os.path.join(output_dir, f'batch_{batch_number}_predictions.pkl')
        
        # 保存当前批次的预测结果
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Batch {batch_number} predictions saved to {output_path}")

## 如果需要将所有预测结果合并，可以在最后将它们合并为一个大的文件
# all_predictions = np.concatenate(all_predictions, axis=0)

# # 最后可以选择保存所有预测结果为一个总文件
# final_output_path = os.path.join(output_dir, 'all_tokenid_predictions.pkl')
# data_to_save_final = {
#     'latents': latents,
#     'true_tokenid': true_tokenids,
#     'predicted_tokenid': all_predictions
# }
# with open(final_output_path, 'wb') as f:
#     pickle.dump(data_to_save_final, f)

# print(f"All predictions saved to {final_output_path}")
 