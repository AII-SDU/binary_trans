# CodeGen - 二进制代码转换工具

CodeGen是一个基于深度学习的二进制代码转换工具，它是PalmTree的修改版本，专注于实现不同架构之间的代码翻译（如ARM到x86）。本项目利用Transformer架构实现了高效的二进制代码分析和转换功能。

## 目录

- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [项目结构](#项目结构)
- [使用流程](#使用流程)
  - [1. 数据生成](#1-数据生成)
  - [2. 编码器训练](#2-编码器训练)
  - [3. 数据集准备](#3-数据集准备)
  - [4. 模型训练](#4-模型训练)
  - [5. 模型测试](#5-模型测试)
- [API使用](#api使用)
- [高级配置](#高级配置)
- [常见问题](#常见问题)

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+（用于GPU加速，推荐）
- Binary Ninja（用于二进制代码分析）
- NetworkX
- Numpy
- Scikit-learn
- tqdm

## 安装说明

1. 克隆仓库：

```bash
git clone 
cd codegen
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. Binary Ninja安装（用于数据生成）：
   - 请访问[Binary Ninja官网](https://binary.ninja/)获取授权并安装
   - 安装Binary Ninja Python API：`pip install binaryninja`

## 项目结构

```
codegen/
├── model.py                 # 序列到序列模型定义
├── eval_utils.py            # 评估和使用工具
├── vocab.py                 # 词汇表处理
├── bleu.py                  # BLEU分数计算
├── config.py                # 配置文件
├── how2use.py               # 使用示例
├── src/                     # 源代码目录
│   ├── train_palmtree.py    # 编码器训练脚本
│   ├── data_generator/      # 数据生成工具
│   │   ├── control_flow_gen.py   # 控制流图生成
│   │   ├── dataflow_gen.py       # 数据流图生成
│   │   └── test.py               # 测试工具
│   └── mkdataset/           # 数据集创建工具
│       └── makedataset.py   # 数据集处理脚本
├── palmtree/                # PalmTree相关代码
├── palmtreemodel/           # 预训练模型存储
└── savemodel/               # 训练模型保存目录
```

## 使用流程

### 1. 数据生成

首先需要生成数据流图(DFG)和控制流图(CFG)用于编码器训练：

```bash
# 创建bin目录用于存放二进制文件
mkdir -p bin
# 将需要分析的二进制文件放入bin目录

# 生成数据流图
cd src/data_generator
python dataflow_gen.py

# 生成控制流图
python control_flow_gen.py
```

这些脚本会分析`bin`目录中的二进制文件，并生成`src/dfg_train.txt`和`src/cfg_train.txt`文件，包含指令序列及其关系。

### 2. 编码器训练

使用生成的数据训练编码器模型：

```bash
cd src
python train_palmtree.py
```

这将使用数据流图和控制流图数据训练PalmTree编码器，结果保存在`data/transformer`目录中。训练过程会自动加载词汇表或根据需要创建新的词汇表。

### 3. 数据集准备

准备用于代码翻译的数据集：

```bash
cd src/mkdataset
python makedataset.py
```

这个脚本处理ARM和x86汇编指令的映射关系，创建一个包含源指令和目标指令的CSV文件，作为翻译模型的训练数据。

### 4. 模型训练

使用生成的数据集训练翻译模型：

```bash
python eval_utils.py --train \
    --train_files assembly_mapping.csv \
    --output_dir savemodel \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 5e-5
```

参数说明：
- `--train_files`: 训练数据文件路径
- `--output_dir`: 模型保存目录
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率

### 5. 模型测试

使用训练好的模型进行代码翻译测试：

```bash
python eval_utils.py --test \
    --test_files test_data.csv \
    --model_path savemodel/best_model \
    --output_file results.csv
```

参数说明：
- `--test_files`: 测试数据文件路径
- `--model_path`: 训练好的模型路径
- `--output_file`: 结果输出文件

翻译结果将保存在指定的输出文件中，并计算BLEU分数评估翻译质量。


## 高级配置

可以通过修改`config.py`调整全局配置：

```python
# 词汇表大小
VOCAB_SIZE = 5000

# 是否使用CUDA
USE_CUDA = True  

# 使用的GPU设备ID
DEVICES = [0]  
CUDA_DEVICE = DEVICES[0]

# 版本号  
VERSION = 1

# 最大序列长度
MAXLEN = 10

# 学习率
LEARNING_RATE = 1e-5
```

