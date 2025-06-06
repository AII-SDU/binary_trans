import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import numpy as np
import palmtree
from palmtree import dataset
from palmtree import trainer
import pickle as pkl
import bert_pytorch
import os

print(palmtree.__file__)

# 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

vocab_path = os.path.join(data_dir, "vocab")
train_cfg_dataset = os.path.join(data_dir, "cfg_train_1.txt")
train_dfg_dataset = os.path.join(data_dir, "dfg_train_1.txt")
test_cfg_dataset = os.path.join(data_dir, "cfg_test_1.txt")
test_dfg_dataset = os.path.join(data_dir, "dfg_test_1.txt")
# sent_dataset = "data/sentence.pkl"
output_path = os.path.join(data_dir, "transformer")

# with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
#     with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
#         vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)
  
# print("VOCAB SIZE:", len(vocab))
# vocab.save_vocab(vocab_path)


print("Loading Vocab", vocab_path)
vocab = dataset.WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))
# print(vocab.itos)


print("Loading Train Dataset")
train_dataset = dataset.BERTDataset(train_cfg_dataset, train_dfg_dataset, vocab, seq_len=20,
                            corpus_lines=None, on_memory=True)

print("Loading Test Dataset")
test_dataset = dataset.BERTDataset(test_cfg_dataset, test_dfg_dataset, vocab, seq_len=20, on_memory=True) 

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=5)



test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=5) \
    if test_dataset is not None else None

print("Building BERT model")
bert = bert_pytorch.BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0)

print("Creating BERT Trainer")
trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0,
                        with_cuda=True, cuda_devices=[0], log_freq=100)


print("Training Start")
for epoch in range(20):
    trainer.train(epoch)
    trainer.save(epoch, output_path)
#    if test_data_loader is not None:
#        trainer.test(epoch)     
