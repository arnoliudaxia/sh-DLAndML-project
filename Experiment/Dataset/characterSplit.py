# 从指定目录读取EEG数据和文本数据，对文本进行符号清理并根据字符数量切分EEG数据，最后将处理后的数据保存到新的目录中。
import os
import pickle
import re

def remove_symbols(text):
    # 使用正则表达式匹配中文字符、字母和数字，去除其他符号
    cleaned_text = re.sub(r'[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]', '', text)
    return cleaned_text


root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/bert-base-chinese/LittlePrince'
output_root_dir = '/home/arno/Projects/EEGDecodingTest/My/Data/qwen-characterSplit'

# 确保输出目录存在
os.makedirs(output_root_dir, exist_ok=True)

for sub_dir in os.listdir(root_dir):
    sub_dir_path = os.path.join(root_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        # 在目标目录中创建子目录
        output_sub_dir = os.path.join(output_root_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(sub_dir_path, file_name)
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    cut_eeg_data, text, _ = loaded_data

                    processed_data = []  # 用于存储处理后的数据

                    for eeg, sen in zip(cut_eeg_data, text):
                        # 移除符号并获取文本长度
                        valid_text = remove_symbols(sen)
                        num_splits = len(valid_text)

                        # 计算每份数据的目标大小
                        total_points = eeg.shape[1]
                        base_size = total_points // num_splits
                        remainder = total_points % num_splits
                        
                        # 创建切分边界
                        split_boundaries = [0]
                        for i in range(num_splits):
                            # 额外点分配到首尾
                            extra = 1 if (i == 0 or i == num_splits - 1) and remainder > 0 else 0
                            split_boundaries.append(split_boundaries[-1] + base_size + extra)
                            remainder -= extra

                        # 按边界切分数据
                        for i in range(num_splits):
                            start, end = split_boundaries[i], split_boundaries[i + 1]
                            segment = eeg[:, start:end]
                            processed_data.append((segment, valid_text[i]))

                    # 保存处理后的数据到新的文件中
                    output_file_path = os.path.join(output_sub_dir, file_name)
                    with open(output_file_path, 'wb') as output_file:
                        pickle.dump(processed_data, output_file)
                        print(f"Processed file saved: {output_file_path}")
