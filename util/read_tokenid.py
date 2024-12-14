import os
import pickle
import numpy as np

# 定义文件夹路径
folder_path = '/media/dell/DATA/BoBo/tokens/tokenid_predictions_old'

# 获取所有 .pkl 文件的路径
pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]

total_correct = 0
total_samples = 0

# 遍历每个文件并计算正确率
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 从数据字典中获取所需数据
    latents = data['latents']            # shape: (N, latent_dim)
    true_tokenid = data['true_tokenid']    # shape: (N, embed_dim)
    predicted_tokenid = data['predicted_tokenid']  # shape: (N, embed_dim)

    N = latents.shape[0]
    correct = 0

    for idx in range(N):
        if true_tokenid[idx] == np.argmax(predicted_tokenid[idx]):
            correct += 1

    total_correct += correct
    total_samples += N
    print(f'pkl_file: {pkl_file}, Accuracy: {correct / N:.4f}')

# 计算总体准确率
overall_accuracy = total_correct / total_samples
print(f'Overall Accuracy: {overall_accuracy:.4f}')
