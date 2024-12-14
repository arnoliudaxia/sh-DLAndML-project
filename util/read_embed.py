import pickle
import numpy as np

# 读取 pkl 文件
pkl_file = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/sub_embed_results/latent_to_embed_model_sub04_predictions.pkl'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# 从数据字典中获取所需数据
latents = data['latents']            # shape: (N, latent_dim)
true_embeds = data['true_embeds']    # shape: (N, embed_dim)
predicted_embeds = data['predicted_embeds']  # shape: (N, embed_dim)

# 随机选择一个索引
N = len(latents)
print(N)
idx = np.random.randint(0, N)

# 打印该索引对应的值和形状
print(f"Index: {idx}")
print("Latent vector:")
print(latents[idx])
print("Shape:", latents[idx].shape)

print("\nTrue embed:")
print(true_embeds[idx])
print("Shape:", true_embeds[idx].shape)

print("\nPredicted embed:")
print(predicted_embeds[idx])
print("Shape:", predicted_embeds[idx].shape)
