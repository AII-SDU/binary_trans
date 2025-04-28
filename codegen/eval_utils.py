"""
工具函数用于评估二进制转换模型
"""
import os
import sys
import re
import argparse
import logging
import random
import datetime
import pickle
import multiprocessing as mp
from itertools import cycle
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

# 添加路径以确保能够正确导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 相对路径配置
BASE_DIR = current_dir
MODEL_DIR = os.path.join(BASE_DIR, "palmtreemodel")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 创建必要的目录
for dir_path in [MODEL_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 相对导入
from config import *
import vocab
from palmtree import dataset
from model import Seq2Seq
from bleu import _bleu

# 设置日志
logger = logging.getLogger(__name__)

# 为同时输出到控制台和文件的工具类
class Writer:
    """同时将输出写入多个输出流的工具类"""
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for writer in self.writers:
            writer.write(text)

    def flush(self):
        for writer in self.writers:
            writer.flush()

def parse_instruction(ins, symbol_map, string_map):
    """
    解析并预处理指令字符串。
    
    参数:
        ins: 指令字符串，例如 "mov, eax, [rax+0x1]"
        symbol_map: 符号映射字典，键为地址，值为符号
        string_map: 字符串映射字典，类似于symbol_map
    
    返回:
        处理后的指令字符串，例如 "mov eax [ rax + 0x1 ]"
    """
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    
    # 处理指令部分
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    
    # 处理操作数部分
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        symbols = [s.strip() for s in symbols if s]
        processed = []
        
        # 处理每个符号
        for j in range(len(symbols)):
            # 检查是否是地址（大于6位但小于15位的十六进制数）
            if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15:
                if int(symbols[j], 16) in symbol_map:
                    processed.append("symbol")
                elif int(symbols[j], 16) in string_map:
                    processed.append("string")
                else:
                    processed.append("address")
            else:
                processed.append(symbols[j])
            processed = [p for p in processed if p]

        token_lst.extend(processed)

    # 返回处理后的标记列表，例如 "mov eax [ rax + 0x1 ]"
    return ' '.join(token_lst)

class TorchVocab(object):
    """
    定义一个词汇表对象，用于将文本数值化。
    
    属性:
        freqs: collections.Counter对象，保存数据中标记的频率
        stoi: collections.defaultdict实例，将标记字符串映射到数字标识符
        itos: 标记字符串列表，按数字标识符索引
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        从collections.Counter创建一个Vocab对象。
        
        参数:
            counter: collections.Counter对象，保存数据中每个值的频率
            max_size: 词汇表的最大大小，None表示无最大值。默认: None
            min_freq: 将标记包含在词汇表中所需的最小频率。小于1的值将设置为1。默认: 1
            specials: 特殊标记列表(如填充或eos)，将添加到词汇表中。默认: ['<pad>']
            vectors: 可用的预训练向量或自定义预训练向量；或上述向量的列表
            unk_init: 默认情况下，将词汇表外的词向量初始化为零向量；可以是任何接受Tensor并返回相同大小Tensor的函数
            vectors_cache: 缓存向量的目录。默认: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # 构建词汇表时不计算特殊标记的频率
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # 先按字母顺序排序，再按频率排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi只是itos的反向字典
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        """重新排序词汇表"""
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        """
        扩展当前词汇表
        
        参数:
            v: 要添加的词汇表
            sort: 是否对添加的词汇表排序
        """
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

class Vocab(TorchVocab):
    """
    扩展的TorchVocab类，添加了特殊标记的索引
    """
    def __init__(self, counter, max_size=None, min_freq=1):
        """
        初始化Vocab对象
        
        参数:
            counter: 计数器对象
            max_size: 词汇表最大大小
            min_freq: 最小词频
        """
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        """将句子转换为序列"""
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        """将序列转换为句子"""
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        """
        加载词汇表
        
        参数:
            vocab_path: 词汇表路径
            
        返回:
            加载的词汇表对象
        """
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        """
        保存词汇表
        
        参数:
            vocab_path: 保存路径
        """
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

class WordVocab(Vocab):
    """
    处理单词的词汇表类
    """
    def __init__(self, texts, max_size=None, min_freq=1):
        """
        初始化WordVocab对象
        
        参数:
            texts: 文本列表
            max_size: 词汇表最大大小
            min_freq: 最小词频
        """
        print("Building Vocab")
        counter = Counter()
        for t in texts:
            for line in tqdm(t):
                if isinstance(line, list):
                    words = line
                else:
                    words = line.replace("\n", " ").replace("\t", " ").split()

                for word in words:
                    counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        """
        将句子转换为序列
        
        参数:
            sentence: 输入句子
            seq_len: 序列长度，如果为None则不进行填充或截断
            with_eos: 是否添加结束标记
            with_sos: 是否添加开始标记
            with_len: 是否返回原始序列长度
            
        返回:
            转换后的序列或序列与原始长度的元组
        """
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        """
        将序列转换为句子
        
        参数:
            seq: 输入序列
            join: 是否将单词列表连接成字符串
            with_pad: 是否保留填充标记
            
        返回:
            单词列表或连接后的字符串
        """
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        """
        加载词汇表
        
        参数:
            vocab_path: 词汇表路径
            
        返回:
            加载的WordVocab对象
        """
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

class UsableTransformer:
    """
    可用于推理的Transformer模型
    """
    def __init__(self, model_path, vocab_path):
        """
        初始化UsableTransformer
        
        参数:
            model_path: 模型路径
            vocab_path: 词汇表路径
        """
        print("Loading Vocab", vocab_path)
        self.vocab = WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        self.model.eval()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)

    def is_torchscript(self):
        """
        检查模型是否是TorchScript模型
        
        返回:
            是否是TorchScript模型的布尔值
        """
        return isinstance(self.model, torch.jit.ScriptModule)

    def encode(self, text, output_option='lst'):
        """
        编码文本
        
        参数:
            text: 要编码的文本
            output_option: 输出选项，'lst'或其他
            
        返回:
            编码后的表示
        """
        segment_label = []
        sequence = []
        
        # 处理输入文本
        for t in text:
            l = (len(t.split(' '))+2) * [1]
            s = self.vocab.to_seq(t)
            s = [3] + s + [2]
            
            # 截断或填充
            if len(l) > 20:
                segment_label.append(l[:20])
            else:
                segment_label.append(l + [0]*(20-len(l)))
            if len(s) > 20:
                sequence.append(s[:20])
            else:
                sequence.append(s + [0]*(20-len(s)))
         
        segment_label = torch.LongTensor(segment_label)
        sequence = torch.LongTensor(sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)

        # 使用模型进行编码
        encoded = self.model.forward(sequence, segment_label)
        word_embeddings = self.model.embedding.token.weight
        print("Encoded output shape:", encoded.shape)
        print("Encoded output shape:", word_embeddings)
        
        # 取平均
        result = torch.mean(encoded.detach(), dim=1)

        del encoded
        if USE_CUDA:
            if output_option == 'numpy':
                return result.data.cpu().numpy()
            else:
                return result.to('cpu')
        else:
            if output_option == 'numpy':
                return result.data.numpy()
            else:
                return result

import pandas as pd
import csv
VOCAB_SIZE = 5000
USE_CUDA = True
DEVICES = [0]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 10
LEARNING_RATE=1e-5

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_example(filename):
    """Read examples from filename."""
    examples=[]
    df = pd.read_csv(filename, keep_default_na=False, na_filter=False)
    
    idx = 0
    for _, row in df.iterrows():
        source_line = row['Source Line']  
        arm = row['ARM Assembly'].split('\n')
        x86 = row['x86 Assembly'].split('\n')
        
        examples.append(Example(idx, x86, arm))
        idx += 1
    return examples
def read_examples(directory):
    """Read examples from all files in a directory."""
    examples=[]
    files = os.listdir(directory)
    for file in files:
        filename = os.path.join(directory, file)
        df = pd.read_csv(filename, keep_default_na=False, na_filter=False)

        idx = 0
        for _, row in df.iterrows():
            source_line = row['Source Line']  
            arm = row['ARM Assembly'].split('\n')
            x86 = row['x86 Assembly'].split('\n')

            examples.append(Example(idx, x86, arm))
            idx += 1
        print(file)
    return examples
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask

def convert_examples_to_features(examples, stage=None):
    """
    将示例转换为模型输入特征
    
    参数:
        examples: 示例列表
        stage: 阶段，'train'或'test'
        
    返回:
        特征列表
    """
    features = []
    vocab = WordVocab.load_vocab(os.path.join(MODEL_DIR, "vocab"))
    
    # 扩展词汇表
    special_tokens = [
        "<eos_p>", "<sos_p>", "cmpb", "b.cc", "b.cs"
    ]
    for token in special_tokens:
        new_vocab = Vocab(Counter({token: 1}))
        vocab.extend(new_vocab)
    
    # 为寄存器和立即数添加标记
    for i in range(34):
        vocab.extend(Vocab(Counter({f"reg{i}": 1})))
        vocab.extend(Vocab(Counter({f"imm{i}": 1})))
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 示例token序列，用于测试词汇表功能
    token_seq = [1, 2, 3, 4, 5, 3048, 3049, 3050, 3070]
    words = vocab.from_seq(token_seq)
    sentence = vocab.from_seq(token_seq, join=True)
    print(f"Token序列示例: {words}")
    print(f"句子示例: {sentence}")
    
    # 处理每个示例
    for example_index, example in enumerate(examples):
        # 处理源序列
        segment_label_source = []
        sequence_source = []
        segment_label_target = []
        sequence_target = []
        
        # 处理源代码(x86汇编)
        for i, instruction in enumerate(example.source):
            # 每个指令的segment label是指令长度+2(开始和结束标记)
            segment_len = len(instruction.split()) + 2
            segment_label = [1] * segment_len
            
            # 将指令转换为token序列，并添加开始和结束标记
            token_seq = vocab.to_seq(instruction)
            token_seq = [3] + token_seq + [2]  # 3=<sos>, 2=<eos>
            
            # 添加序列位置标记
            if i == 0:  # 第一个指令
                token_seq = [3049] + token_seq  # <sos_p>
                segment_label = [1] * (len(segment_label) + 1)
            if i == len(example.source) - 1:  # 最后一个指令
                token_seq = token_seq + [3048]  # <eos_p>
                segment_label = [1] * (len(segment_label) + 1)
            
            # 截断或填充到最大长度20
            if len(segment_label) > 20:
                segment_label_source.append(segment_label[:20])
            else:
                segment_label_source.append(segment_label + [0] * (20 - len(segment_label)))
                
            if len(token_seq) > 20:
                sequence_source.append(token_seq[:20])
            else:
                sequence_source.append(token_seq + [0] * (20 - len(token_seq)))
        
        # 转换为张量
        segment_label_source = torch.LongTensor(segment_label_source)
        sequence_source = torch.LongTensor(sequence_source)
        source_ids = sequence_source
        source_mask = segment_label_source
        
        # 处理目标代码(ARM汇编)
        if stage == "test":
            target_tokens = []
            segment_label_target = torch.LongTensor([])
            sequence_target = torch.LongTensor([])
        else:
            # 与处理源代码类似
            for i, instruction in enumerate(example.target):
                segment_len = len(instruction.split()) + 2
                segment_label = [1] * segment_len
                
                token_seq = vocab.to_seq(instruction)
                token_seq = [3] + token_seq + [2]
                
                if i == 0:
                    token_seq = [3049] + token_seq
                    segment_label = [1] * (len(segment_label) + 1)
                if i == len(example.target) - 1:
                    token_seq = token_seq + [3048]
                    segment_label = [1] * (len(segment_label) + 1)
                
                if len(segment_label) > 20:
                    segment_label_target.append(segment_label[:20])
                else:
                    segment_label_target.append(segment_label + [0] * (20 - len(segment_label)))
                    
                if len(token_seq) > 20:
                    sequence_target.append(token_seq[:20])
                else:
                    sequence_target.append(token_seq + [0] * (20 - len(token_seq)))
                    
            segment_label_target = torch.LongTensor(segment_label_target)
            sequence_target = torch.LongTensor(sequence_target)
            
        target_ids = sequence_target
        target_mask = segment_label_target
        
        # 创建特征对象
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    
    return features

def set_seed(seed=42):
    """
    设置随机种子以确保结果可重现
    
    参数:
        seed: 随机种子，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")

def main():
    """主函数，处理训练、评估和测试过程"""
    # 设置日志输出到控制台和文件
    console = sys.stdout
    log_file = os.path.join(LOG_DIR, 'eval_utils.log')
    file = open(log_file, 'w')
    sys.stdout = Writer(console, file)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="二进制代码转换模型训练和评估工具")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str,
                        help="模型预测和检查点的输出目录")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否在开发集上进行评估")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="加载训练好的模型的路径，应包含.bin文件")
    parser.add_argument("--do_test", action='store_true',
                        help="是否在测试集上进行评估")
    parser.add_argument("--vocab_path", default=os.path.join(MODEL_DIR, "vocab"), type=str,
                        help="词汇表文件路径")
    parser.add_argument("--model_path", default=os.path.join(MODEL_DIR, "transformer.ep19"), type=str,
                        help="预训练模型路径")
    parser.add_argument("--test_file", default=os.path.join(DATA_DIR, "test.csv"), type=str,
                        help="测试文件路径")
    parser.add_argument("--train_dir", default=os.path.join(DATA_DIR, "train"), type=str,
                        help="包含训练文件的目录")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="批处理大小")
    parser.add_argument("--epochs", default=100, type=int,
                        help="训练轮数")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="学习率")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    parser.add_argument("--max_length", default=10, type=int,
                        help="最大序列长度")
    parser.add_argument("--min_length", default=1, type=int,
                        help="最小序列长度")
    parser.add_argument("--eval_steps", default=1000, type=int,
                        help="多少步进行一次评估")
    
    args = parser.parse_args()
    logger.info(args)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载预训练的PalmTree模型
    print(f"正在加载预训练模型: {args.model_path}")
    print(f"正在加载词汇表: {args.vocab_path}")
    
    palmtree = UsableTransformer(model_path=args.model_path, vocab_path=args.vocab_path)
    vocab_size = len(palmtree.vocab)
    print(f"原始词汇表大小: {vocab_size}")

    # 扩展词汇表
    special_tokens = {
        "<eos_p>": "序列结束标记",
        "<sos_p>": "序列开始标记",
        "cmpb": "比较字节指令",
        "b.cc": "条件分支指令(cc)",
        "b.cs": "条件分支指令(cs)"
    }
    
    for token, desc in special_tokens.items():
        new_vocab = Vocab(Counter({token: 1}))
        palmtree.vocab.extend(new_vocab)
        print(f"添加特殊标记 '{token}': {desc}")
    
    # 为寄存器和立即数添加特殊标记
    for i in range(34):
        palmtree.vocab.extend(Vocab(Counter({f"reg{i}": 1})))
        palmtree.vocab.extend(Vocab(Counter({f"imm{i}": 1})))
    
    vocab_size = len(palmtree.vocab)
    print(f"扩展后词汇表大小: {vocab_size}")
    
    # 测试词汇表功能
    token_seq = [1, 2, 3, 4, 5, 3048, 3049, 3050, 3070]
    words = palmtree.vocab.from_seq(token_seq)
    sentence = palmtree.vocab.from_seq(token_seq, join=True)
    print(f"词汇表测试 - Tokens: {words}")
    print(f"词汇表测试 - 句子: {sentence}")

    # 初始化设备
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化解码器
    print("初始化Transformer解码器")
    decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=8)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    decoder = decoder.to(device)

    # 处理嵌入权重
    weight = palmtree.model.embedding.token.weight
    weight = weight.to(device)
    embed_dim = weight.size(1)
    print(f"嵌入维度: {embed_dim}")

    # 扩展嵌入权重以匹配扩展的词汇表
    print("扩展嵌入权重...")
    for _ in range(73):  # 添加额外权重以匹配扩展的词汇表
        new_weight = torch.cat([weight, torch.randn(1, embed_dim).to(device)])
        weight = new_weight

    palmtree.model.embedding.token.weight = nn.Parameter(weight)
    print(f"嵌入权重形状: {palmtree.model.embedding.token.weight.shape}")
    
    # 初始化序列到序列模型
    print("初始化Seq2Seq模型")
    model = Seq2Seq(
        encoder=palmtree.model,
        decoder=decoder,
        vocab_size=vocab_size,
        beam_size=10,
        max_length=200,
        sos_id=3049,  # <sos_p>
        eos_id=3048   # <eos_p>
    )
    
    # 加载模型权重（如果提供）
    if args.load_model_path is not None:
        print(f"从{args.load_model_path}加载模型...")
        model.load_state_dict(torch.load(args.load_model_path))

    # 将模型移动到设备
    model.to(device)
    
    # 训练模式
    if args.do_train:
        print("\n" + "="*50)
        print("开始训练过程")
        print("="*50)
        
        print(f"从{args.train_dir}加载训练数据...")
        examples = read_examples(args.train_dir)
        train_examples = examples
        print(f"加载了{len(train_examples)}个训练示例")
        
        print("将示例转换为特征...")
        train_features = convert_examples_to_features(train_examples)
        
        # 过滤掉长度不符合要求的特征
        print(f"过滤特征 (最大长度: {args.max_length}, 最小长度: {args.min_length})...")
        filtered_features = [
            f for f in train_features 
            if (len(f.source_ids) <= args.max_length and len(f.source_ids) >= args.min_length and
                len(f.source_mask) <= args.max_length and len(f.source_mask) >= args.min_length and   
                len(f.target_ids) <= args.max_length and len(f.target_ids) >= args.min_length and
                len(f.target_mask) <= args.max_length and len(f.target_mask) >= args.min_length)
        ]
        print(f"原始特征数量: {len(train_features)}")
        print(f"过滤后的特征数量: {len(filtered_features)}")
        train_features = filtered_features
        
        # 填充三维特征
        print("填充特征到统一大小...")
        padded_features = []
        for f in train_features:
            
            padded_feature = {
                'source_ids': torch.cat([f.source_ids, torch.zeros(args.max_length - f.source_ids.shape[0], f.source_ids.shape[1], dtype=torch.long)]),
                'source_mask': torch.cat([f.source_mask, torch.zeros(args.max_length - f.source_mask.shape[0], f.source_mask.shape[1], dtype=torch.long)]),
                'target_ids': torch.cat([f.target_ids, torch.zeros(args.max_length - f.target_ids.shape[0], f.target_ids.shape[1], dtype=torch.long)]),
                'target_mask': torch.cat([f.target_mask, torch.zeros(args.max_length - f.target_mask.shape[0], f.target_mask.shape[1], dtype=torch.long)])
            }
            padded_features.append(padded_feature)

        train_features = padded_features

        all_source_ids = torch.stack([f['source_ids'] for f in train_features])
        all_source_mask = torch.stack([f['source_mask'] for f in train_features])
        all_target_ids = torch.stack([f['target_ids'] for f in train_features])
        all_target_mask = torch.stack([f['target_mask'] for f in train_features]) 
        # print(all_source_ids.shape)
        print("target_ids:", all_target_ids[0])
        print("target_mask:", all_target_mask[0])
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=256,num_workers=4)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        num_train_optimization_steps = args.epochs * len(train_dataloader)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-5},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.batch_size*0.1,
                                                    num_training_steps=len(train_dataloader)*args.batch_size)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", args.batch_size)
        print("  Num epoch = %d", num_train_optimization_steps)
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        bar = range(num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            gradient_accumulation_steps = 1
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            loss,_,_ = model(source_ids=source_ids.long(),source_mask=source_mask.long(),target_ids=target_ids.long(),target_mask=target_mask.long())
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*gradient_accumulation_steps/(nb_tr_steps+1),4)
            if (global_step + 1)%50==0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(current_time)
                print("  step {} loss {}".format(global_step + 1,train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            # del loss
            torch.cuda.empty_cache()
            if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    eval_flag = True

            # if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
            if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
                
                # preds = model(source_ids=source_ids.long(),source_mask=source_mask.long())  
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_example(args.test_file)
                    eval_features = convert_examples_to_features(eval_examples)
                    eval_filtered_features = [
                        f for f in eval_features if len(f.source_ids) <= args.max_length and
                        len(f.source_ids) >= args.min_length
                        and len(f.source_mask) <= args.max_length
                        and len(f.source_mask) >= args.min_length   
                        and len(f.target_ids) <= args.max_length
                        and len(f.target_ids) >= args.min_length
                        and len(f.target_mask) <= args.max_length 
                        and len(f.target_mask) >= args.min_length
                        ]
                    print("原始eval特征数量:", len(eval_features))
                    print("过滤后的eval特征数量:", len(eval_filtered_features))
                    eval_features = eval_filtered_features
                    # 填充三维特征
                    eval_padded_features = []
                    for f in eval_features:
                        eval_padded_feature = {
                        'source_ids': torch.cat([f.source_ids, torch.zeros(args.max_length - f.source_ids.shape[0], f.source_ids.shape[1])]),
                        'source_mask': torch.cat([f.source_mask, torch.zeros(args.max_length - f.source_mask.shape[0], f.source_mask.shape[1])]),
                        'target_ids': torch.cat([f.target_ids, torch.zeros(args.max_length - f.target_ids.shape[0], f.target_ids.shape[1])]),
                        'target_mask': torch.cat([f.target_mask, torch.zeros(args.max_length - f.target_mask.shape[0], f.target_mask.shape[1])])
                        }
                        eval_padded_features.append(eval_padded_feature)
                    eval_features = eval_padded_features

                    all_source_ids = torch.stack([f['source_ids'] for f in eval_features])
                    all_source_mask = torch.stack([f['source_mask'] for f in eval_features])
                    all_target_ids = torch.stack([f['target_ids'] for f in eval_features])
                    all_target_mask = torch.stack([f['target_mask'] for f in eval_features]) 
                    # print(all_source_ids.shape)
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
                
                print("\n***** Running evaluation *****")
                print("  Num examples = %d", len(eval_examples))
                print("  Batch size = %d", args.batch_size)
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids.long(),source_mask=source_mask.long(),
                                           target_ids=target_ids.long(),target_mask=target_mask.long())     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()

                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  "+"*"*20) 



                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    print("  Best ppl:%s",round(np.exp(eval_loss),5))
                    print("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  
                output_file_path = os.path.join(args.output_dir, "result_eval.csv")
                if os.path.exists(output_file_path):
                    os.remove(output_file_path)
                    print("DELETE")
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_example(args.test_file)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples)
                    eval_filtered_features = [
                        f for f in eval_features if len(f.source_ids) <= args.max_length 
                        and len(f.source_ids) >= args.min_length
                        and len(f.source_mask) <= args.max_length
                        and len(f.source_mask) >= args.min_length   
                        and len(f.target_ids) <= args.max_length
                        and len(f.target_ids) >= args.min_length
                        and len(f.target_mask) <= args.max_length 
                        and len(f.target_mask) >= args.min_length
                        ]
                    print("原始eval特征数量:", len(eval_features))
                    print("过滤后的eval特征数量:", len(eval_filtered_features))    
                    eval_features = eval_filtered_features
                    eval_padded_features = []
                    for f in eval_features:
                        eval_padded_feature = {
                            'source_ids': torch.cat([f.source_ids, torch.zeros(args.max_length - f.source_ids.shape[0], f.source_ids.shape[1], dtype=torch.long)]),
                            'source_mask': torch.cat([f.source_mask, torch.zeros(args.max_length - f.source_mask.shape[0], f.source_mask.shape[1], dtype=torch.long)]),
                            'target_ids': torch.cat([f.target_ids, torch.zeros(args.max_length - f.target_ids.shape[0], f.target_ids.shape[1], dtype=torch.long)]),
                            'target_mask': torch.cat([f.target_mask, torch.zeros(args.max_length - f.target_mask.shape[0], f.target_mask.shape[1], dtype=torch.long)])
                        }
                        eval_padded_features.append(eval_padded_feature)
                    eval_features = eval_padded_features
                    all_source_ids = torch.stack([f['source_ids'] for f in eval_features])
                    all_source_mask = torch.stack([f['source_mask'] for f in eval_features])
                    all_target_ids = torch.stack([f['target_ids'] for f in eval_features])
                    all_target_mask = torch.stack([f['target_mask'] for f in eval_features]) 
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                            # Calculate bleu
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

                model.eval() 
                p=[]
                rows = []
                for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch                    
                    with torch.no_grad():
                        # print("source_ids",source_ids)
                        # print("source_mask",source_mask)
                        preds = model(source_ids=source_ids,source_mask=source_mask)
                        # print("preds.shape",preds.shape)
                        i =0
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t= t[:t.index(0)]
                            text = t
                            source =[]
                            target =[]
                            for n in range(source_ids.shape[1]):
                                valid_len = source_mask[i][n].sum()
                                valid_out = source_ids[i][n][:valid_len].long()
                                valid_out = valid_out.tolist()
                                source.extend(valid_out)

                                target_valid_len = target_mask[i][n].sum()
                                target_valid_out = target_ids[i][n][:target_valid_len].long()
                                target_valid_out = target_valid_out.tolist()
                                target.extend(target_valid_out)

                            i=i+1
                            text = palmtree.vocab.from_seq(t, join=True)
                            source = palmtree.vocab.from_seq(source, join=True)
                            target = palmtree.vocab.from_seq(target, join=True)

                            text = text.replace('<eos>', '\n')
                            text = text.replace('<sos_p>', '')
                            text = text.replace('<sos>', '')
                            text = text.replace('<eos_p>', '')
                            source = source.replace('<eos>', '\n')
                            source = source.replace('<sos_p>', '') 
                            source = source.replace('<sos>', '')
                            source = source.replace('<eos_p>', '')
                            target = target.replace('<eos>', '\n')
                            target = target.replace('<sos_p>', '') 
                            target = target.replace('<sos>', '')
                            target = target.replace('<eos_p>', '')
                            if text=='':
                                text= 1
                                print("change text to 1")

                            rows.append({'source': source, 'target': target, 'text': text})
                            p.append(text)
                        df = pd.DataFrame(rows)
                        df.to_csv(output_file_path, index=False)

                model.train()
                dev_bleu=round(_bleu(output_file_path))
                if dev_bleu>=best_bleu:
                    print("  Best bleu:%s",dev_bleu)
                    print("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                # print("preds开始")

                # model.eval() 
                # p=[]
                # for batch in eval_dataloader:
                #     batch = tuple(t.to(device) for t in batch)
                #     source_ids,source_mask= batch                  
                #     with torch.no_grad():
                #         preds = model(source_ids=source_ids,source_mask=source_mask)  
                #         for pred in preds:
                #             t=pred[0].cpu().numpy()
                #             t=list(t)
                #             if 0 in t:
                #                 t=t[:t.index(0)]
                #             text = palmtree.vocab.from_seq(t, join=True)
                #             p.append(text)
                # model.train()
                # predictions=[]
                # accs=[]
                # with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                #     for ref,gold in zip(p,eval_examples):
                #         predictions.append(str(gold.idx)+'\t'+ref)
                #         f.write(ref+'\n')
                #         f1.write(gold.target+'\n')     
                #         accs.append(ref==gold.target)
    if args.do_test:
        print("\n" + "="*50)
        print("开始测试过程")
        print("="*50)
        
        # 如果提供了测试文件，则使用它
        files = []
        files.append(args.test_file)
        
        # 处理每个测试文件
        for idx, file in enumerate(files):   
            print(f"测试文件: {file}")
            
            # 加载测试数据
            print("加载测试示例...")
            eval_examples = read_example(file)
            print(f"加载了{len(eval_examples)}个测试示例")
            
            # 转换为特征
            print("将测试示例转换为特征...")
            eval_features = convert_examples_to_features(eval_examples)
            
            # 过滤特征
            max_length = args.max_length
            min_length = args.min_length
            print(f"过滤特征 (最大长度: {max_length}, 最小长度: {min_length})...")
            eval_filtered_features = [
                f for f in eval_features if len(f.source_ids) <= max_length 
                and len(f.source_ids) >= min_length
                and len(f.source_mask) <= max_length
                and len(f.source_mask) >= min_length   
                and len(f.target_ids) <= max_length
                and len(f.target_ids) >= min_length
                and len(f.target_mask) <= max_length 
                and len(f.target_mask) >= min_length
            ]
            print(f"原始测试特征数量: {len(eval_features)}")
            print(f"过滤后的测试特征数量: {len(eval_filtered_features)}")    
            eval_features = eval_filtered_features
            
            # 填充特征
            print("填充特征到统一大小...")
            eval_padded_features = []
            for f in eval_features:
                eval_padded_feature = {
                    'source_ids': torch.cat([f.source_ids, torch.zeros(max_length - f.source_ids.shape[0], f.source_ids.shape[1], dtype=torch.long)]),
                    'source_mask': torch.cat([f.source_mask, torch.zeros(max_length - f.source_mask.shape[0], f.source_mask.shape[1], dtype=torch.long)]),
                    'target_ids': torch.cat([f.target_ids, torch.zeros(max_length - f.target_ids.shape[0], f.target_ids.shape[1], dtype=torch.long)]),
                    'target_mask': torch.cat([f.target_mask, torch.zeros(max_length - f.target_mask.shape[0], f.target_mask.shape[1], dtype=torch.long)])
                }
                eval_padded_features.append(eval_padded_feature)
            eval_features = eval_padded_features

            # 创建数据集
            print("创建测试数据集...")
            all_source_ids = torch.stack([f['source_ids'] for f in eval_features])
            all_source_mask = torch.stack([f['source_mask'] for f in eval_features])
            all_target_ids = torch.stack([f['target_ids'] for f in eval_features])
            all_target_mask = torch.stack([f['target_mask'] for f in eval_features]) 
            eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)   

            # 创建数据加载器
            print("创建测试数据加载器...")
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

            # 进行测试推理
            print("开始测试推理...")
            model.eval() 
            predictions = []
            rows = []
            
            # 准备输出文件路径
            output_file_path = os.path.join(args.output_dir, f"result_test_{idx}.csv")
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
                print(f"删除已存在的输出文件: {output_file_path}")
                
            # 遍历批次进行预测
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="测试推理进度"):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                
                with torch.no_grad():
                    # 使用模型预测
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    print(f"预测形状: {preds.shape}")
                    
                    # 处理每个预测结果
                    i = 0
                    for pred in preds:
                        # 获取预测的token序列
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]  # 截断到第一个0，即序列结束符
                            
                        # 提取源和目标序列
                        source = []
                        target = []
                        for n in range(source_ids.shape[1]):
                            # 处理源序列
                            valid_len = source_mask[i][n].sum()
                            valid_out = source_ids[i][n][:valid_len].long()
                            valid_out = valid_out.tolist()
                            source.extend(valid_out)

                            # 处理目标序列
                            target_valid_len = target_mask[i][n].sum()
                            target_valid_out = target_ids[i][n][:target_valid_len].long()
                            target_valid_out = target_valid_out.tolist()
                            target.extend(target_valid_out)

                        i += 1
                        
                        # 将token序列转换回文本
                        text = palmtree.vocab.from_seq(t, join=True)
                        source = palmtree.vocab.from_seq(source, join=True)
                        target = palmtree.vocab.from_seq(target, join=True)
                        
                        # 清理特殊标记
                        special_tokens = ['<eos>', '<sos_p>', '<sos>', '<eos_p>']
                        for token in special_tokens:
                            text = text.replace(token, '\n' if token == '<eos>' else '')
                            source = source.replace(token, '\n' if token == '<eos>' else '')
                            target = target.replace(token, '\n' if token == '<eos>' else '')
                        
                        # 处理空预测
                        if text == '':
                            text = 1
                            print("预测为空，使用默认值1代替")

                        # 保存结果
                        rows.append({'source': source, 'target': target, 'text': text})
                        predictions.append(text)
                
                # 阶段性保存结果
                if len(rows) % 100 == 0 and len(rows) > 0:
                    df = pd.DataFrame(rows)
                    df.to_csv(output_file_path, index=False)
                    print(f"已保存{len(rows)}个结果到{output_file_path}")
            
            # 保存最终结果
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_file_path, index=False)
                print(f"已保存所有{len(rows)}个结果到{output_file_path}")

            # 计算BLEU分数
            try:
                bleu_score = round(_bleu(output_file_path))
                print(f"BLEU-4 分数: {bleu_score}")
            except Exception as e:
                print(f"计算BLEU分数时出错: {e}")
            
            # 恢复模型训练模式
            model.train()
            
    # 关闭日志文件
    print("处理完成")
    file.close()
if __name__ == "__main__":
    main()
    