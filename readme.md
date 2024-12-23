# EEG decoding to language V0.1

## 输入输出

原始数据集：人读好多句话。每一句话采集脑电信号，包含128通道，不定长度样本点。

1. 首先我将一句话按照字数切分，得到每个字和EEG信号对应。
2. 然后我训练了一个EEG信号的autoencoder，将shape (128, t)的多通道信号压缩到(1,64)，得到和一个文字字对应的latent变量
3. 然后我又利用qwen2-7b（阿里巴巴通义千问模型）的tokenizer，将文字字转化为token，得到最终的pairs (token, latent)，其中token的shape为(1,)，是一个数字；latent的shape为(1,64)，是一个向量。

数据集文件类似于`Data/qwen-characterSplit/sub04/run_1_with_latent_tokenid.pkl`，读取的代码在`util/getAllData.py`中

输出：“重建出”一个中文字。

运行命令遵循下面的模式

```
PYTHONPATH=$(pwd) python Experiment/Dataset/autoencoder4EEG-inference.py
```

## 结构目录

| 路径                             | 说明                            |
|----------------------------------|---------------------------------|
| readme.md                        | 说明文件                        |
| Data/qwen-characterSplit/     | 数据集                          |
| Model/dataloader.py           | PyTorch格式的Dataloader（注意过滤文件名） |
| util                          | 包含一些（可能）有用的函数      |

## 核心实验

- `My/Experiment/Dataset` 涉及对于数据的一些处理算法
- `My/Experiment/AutoEncoder/autoencoder4EEG.py` EEGencoder模型，用来获取一个低纬度的latent表示

获取数据，注意修改
```
Chinese_reading_task_eeg_processing/data_preprocessing_and_alignment/align_eeg_with_sentence.py
```

训练autoencoder
```
PYTHONPATH=$(pwd) python My/Experiment/AutoEncoder/autoencoder4EEG.py --batchSize 4096 --SaveModelPath My/Model/AutoEncoder/mask8_12_6 --UseWandb --lr 1e-4 --mask 8 12 6
```

使用训练好的autoencoder在test subjet上inference
```
PYTHONPATH=$(pwd) python My/Experiment/Dataset/autoencoder4EEG-inference.py --model_path <Trained Model Path>
```

使用autoencoder inference获取test subjet latent之后，训练 text embedding的映射 
```
 PYTHONPATH=$(pwd) python My/Experiment/latent2Embedding/latent2embed.py --SaveModelPath My/Model/latent2Embedding  --lr 1e-4 --UseWandb
```


## 总体思想（不用看）

1. **EEG 信号预处理：** 
   - 首先对 EEG 数据进行预处理，包括滤波、去除噪声等步骤，以提取高质量的信号。
   - 接下来，将 EEG 信号分段，保证与对应语言刺激的数据同步，得到每个时间段的EEG特征。

2. **EEG 与 BERT 输入嵌入空间的映射：**
   - 你需要构建一个深度神经网络模型，将 EEG 特征映射到 BERT 的输入嵌入空间。
   - 使用一个编码器（Encoder）网络，学习将 EEG 信号特征编码到与 BERT 输入嵌入空间相匹配的维度。这可以是一个前馈神经网络（如多层感知机 MLP），或者更加复杂的结构，如卷积神经网络（CNN）或循环神经网络（RNN）来捕获时间依赖关系。

3. **损失函数设计：**
   - 损失函数可以考虑使用余弦相似度（cosine similarity）损失，确保 EEG 特征与语言嵌入在相似的向量空间中。
   - 你也可以利用对比学习（contrastive learning）的方式，确保来自相同语言刺激的 EEG 特征和 BERT 嵌入尽可能相似，而不同刺激的特征尽可能不同。

4. **训练过程：**
   - 需要有配对的数据，即每段 EEG 信号都有一个相应的语言刺激。你可以使用这种配对的数据来监督训练你的网络，将 EEG 信号学习到 BERT 的输入嵌入空间中。
   - 在训练过程中，输入 EEG 信号，通过你的网络生成一个向量，然后计算这个向量与对应的 BERT 嵌入之间的相似性。

5. **语言解码：**
   - 一旦训练完成，可以将新录制的 EEG 信号输入到你的网络中，生成相应的 BERT 嵌入。
   - 然后通过使用 BERT 模型（或其他解码机制）将这些嵌入转换回文本，生成最有可能的语言刺激。

6. **评估：**
   - 你可以通过 BLEU、ROUGE 等自然语言生成指标评估模型解码的语言输出的质量。
   - 还可以使用分类准确率或相似性指标评估EEG信号与语言嵌入映射的效果。


