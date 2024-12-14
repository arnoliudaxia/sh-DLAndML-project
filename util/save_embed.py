import torch
import pickle
import os
from latent2embed import LatentToEmbedModel, load_data


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载目录和模型目录
root_dir = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/Data/qwen-characterSplit'
model_dir = r'/media/dell/DATA/BoBo/sub_embed_3_models'
output_dir = r'/media/dell/DATA/BoBo/sh-DLAndML-project-master/sub_embed_3_results'

# 若输出目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载数据
latents, embeds = load_data(root_dir)
latents_tensor = torch.tensor(latents, dtype=torch.float32).to(device)

# 遍历模型目录下的所有pth文件
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"Loading model: {model_path}")

    # 初始化模型并加载权重
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
    output_filename = os.path.splitext(model_file)[0] + '_predictions.pkl'
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Predictions for {model_file} saved to {output_path}")
