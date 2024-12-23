import torch
import pickle
import os
from My.Experiment.latent2Embedding.latent2embed import LatentToEmbedModel, load_data, load_data_for_subs


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载目录和模型目录
root_dir = "My/Data/qwen-characterSplit"
model_path = "My/Model/latent2Embedding/latent_to_embed_model_sub08.pth"
output_dir = "My/Data/yhw/mask8"
# 若输出目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载数据
_, _, latents, embeds = load_data_for_subs(root_dir, "sub08")
latents_tensor = torch.tensor(latents, dtype=torch.float32).to(device)


print(f"Loading model: {model_path}")
model = LatentToEmbedModel().to(device)
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

# 根据模型文件名生成对应的输出pkl文件名
output_filename =  'latentEmbedding_predictions.pkl'
output_path = os.path.join(output_dir, output_filename)

with open(output_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Predictions for {model_path} saved to {output_path}")
