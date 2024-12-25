import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str)
# parser.add_argument('--output_path', type=str)
parser.add_argument('--mask', type=int, nargs='+', help="Mask掉的subject（不用做训练）")
args = parser.parse_args()
print(args)

model_name ='My/LLM/Qwen2.5-7B-Instruct'
device="cuda"
    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def getEmbedding(text, raw=False):
        if not raw:
                messages = [
                        # {"role": "system", "content": ' You are a helpful assistant. The user has a language impairment, and their expressions may contain a lot of noise. You need to rephrase their meaning clearly.'},
                        {"role": "system", "content": ' You are a helpful assistant.'},
                        {"role": "user", "content": text}
                        ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        return model.get_input_embeddings()(model_inputs.input_ids)

#region 已经训练过一个模型从EEG latent预测embedding，读取所有的（character，训练embed）
root_dir = 'Data/qwen-characterSplit'
sub=args.mask # ! Mask

characters=[]
for sub_dir in os.listdir(root_dir):
    if all(item not in sub_dir for item in map(str, sub)):
        continue
    else:
    # if "08" in sub_dir:
    # if "04" in sub_dir or "05" in sub_dir or "08" in sub_dir or True:
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.pkl') and 'latent' in file_name and '_tokenid'  in file_name and'_tokenEmbedding' in file_name:
                    file_path = os.path.join(sub_dir_path, file_name)
                    with open(file_path, 'rb') as file:
                        loaded_data = pickle.load(file)
                    # 如果loaded_data里面每一个元素没有下面那么多成分，以每个的shape为准，shape总是对的。比如没有一个3584长度的东西说明embedding没在里面
                    for  eeg, character, latent, tokenid, embed in loaded_data:
                        # eeg.shape (1, 128, 256) 多通道EEG信号
                        # character -> str 一个中文汉字
                        # latent.shape (1, 64) EEG压缩后的向量
                        # tokenid array([24]) 汉字对应的token
                        # embed.shape (3584,)  词嵌入向量
                        characters.append([character,embed])
                       
print(f"预测样本数量：{len(characters)}")
#endregion