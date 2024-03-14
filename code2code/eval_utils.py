from torch.autograd import Variable
import torch
import re
import numpy
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from config import *
import vocab
from palmtree import dataset
import pickle
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp

from model import Seq2Seq
from bleu import _bleu

import os
import argparse
import logging
from itertools import cycle
import random
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
# this function is how I parse and pre-pocess instructions for palmtree. It is very simple and based on regular expressions. 
# If I use IDA pro or angr instead of Binary Ninja, I would have come up with a better solution.

logger = logging.getLogger(__name__)

def parse_instruction(ins, symbol_map, string_map):
    # arguments:
    # ins: string e.g. "mov, eax, [rax+0x1]"
    # symbol_map: a dict that contains symbols the key is the address and the value is the symbol 
    # string_map : same as symbol_map in Binary Ninja, constant strings will be included into string_map 
    #              and the other meaningful strings like function names will be included into the symbol_map
    #              I think you do not have to separate them. This is just one of the possible nomailization stretagies.
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    for i in range(len(operand)):
        # print(operand)
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        symbols = [s.strip() for s in symbols if s]
        processed = []
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15: 
                # I make a very dumb rule here to treat number larger than 6 but smaller than 15 digits as addresses, 
                # the others are constant numbers and will not be normalized.
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

    # the output will be like "mov eax [ rax + 0x1 ]"
    return ' '.join(token_lst)

class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
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
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for t in texts:
            for line in tqdm.tqdm(t):
                if isinstance(line, list):
                    words = line
                else:
                    words = line.replace("\n", " ").replace("\t", " ").split()

                for word in words:
                    counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
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
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

class UsableTransformer:
    
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        self.vocab = WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        self.model.eval()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)

    def is_torchscript(self):
        return isinstance(self.model, torch.jit.ScriptModule)

    def encode(self, text, output_option='lst'):

        segment_label = []
        sequence = []
        for t in text:
            l = (len(t.split(' '))+2) * [1]
            s = self.vocab.to_seq(t)
            s = [3] + s + [2]
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

        encoded = self.model.forward(sequence, segment_label)
        word_embeddings = self.model.embedding.token.weight
        print("Encoded output shape:", encoded.shape)
        print("Encoded output shape:", word_embeddings)
        result = torch.mean(encoded.detach(), dim=1)

        del encoded
        if USE_CUDA:
            if numpy:
                return result.data.cpu().numpy()
            else:
                return result.to('cpu')
        else:
            if numpy:
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

def convert_examples_to_features(examples,stage=None):
    features = []
    vocab = WordVocab.load_vocab("/home/kingdom/PalmTree-master/pre-trained_model/palmtreemodel/vocab")
    for example_index, example in enumerate(examples):
        #source
        segment_label_source = []
        sequence_source = []
        segment_label_target = []
        sequence_target = []
        for i, t in enumerate(example.source):
            l = (len(t.split())+2) * [1]
            s = vocab.to_seq(t)
            # print(t, s)
            s = [3] + s + [2]
            if i == 0:
                s = [4684] + s
                l = (len(l)+1) * [1]
            if i == len(example.source)-1:
                s = s + [4683]
                l = (len(l)+1) * [1]
            # if(len(s)>7 and s[1]==22 and s[6]==562):
            #     print(t)
            #     print(t.split())
            #     print(len(t.split())+2)
            if len(l) > 20:
                segment_label_source.append(l[:20])
            else:
                segment_label_source.append(l + [0]*(20-len(l)))
            if len(s) > 20:
                sequence_source.append(s[:20])
            else:
                sequence_source.append(s + [0]*(20-len(s)))
         
        segment_label_source = torch.LongTensor(segment_label_source)
        sequence_source = torch.LongTensor(sequence_source)
        source_ids = sequence_source
        source_mask = segment_label_source
 
        #target
        if stage=="test":
            # target_tokens = tokenizer.tokenize("None")
            target_tokens = []
        else:
             for i, t in enumerate(example.target):
                l = (len(t.split())+2) * [1]
                s = vocab.to_seq(t)
                # print(t, s)
                s = [3] + s + [2]
                if i == 0:
                    s = [4684] + s
                    l = (len(l)+1) * [1]
                if i == len(example.target)-1:
                    s = s + [4683]
                    l = (len(l)+1) * [1]
                if len(l) > 20:
                    segment_label_target.append(l[:20])
                else:
                    segment_label_target.append(l + [0]*(20-len(l)))
                if len(s) > 20:
                    sequence_target.append(s[:20])
                else:
                    sequence_target.append(s + [0]*(20-len(s)))
        segment_label_target = torch.LongTensor(segment_label_target)
        sequence_target = torch.LongTensor(sequence_target)
        target_ids = sequence_target
        target_mask = segment_label_target
        
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

def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    args = parser.parse_args()
    logger.info(args)
    # for e in examples[:5]:
    #     print(f'idx: {e.idx}') 
    #     print(f'x86: {e.source}')
    #     print(f'arm: {e.target}')
    #     print('-' * 50)
    # features = convert_examples_to_features(examples[:5])
    # for e in features:
    #     print(f'example_index: {e}') 
    #     print(f'source_ids: {e.source_ids}')
    #     print(f'target_ids: {e.target_ids}')
    #     print('-' * 50)
    palmtree = UsableTransformer(model_path="/home/kingdom/PalmTree-master/pre-trained_model/palmtreemodel/transformer.ep19", vocab_path="/home/kingdom/PalmTree-master/pre-trained_model/palmtreemodel/vocab")
    vocab=len(palmtree.vocab)

    new_vocab = Vocab(Counter({"<eos_p>": 1}))
    palmtree.vocab.extend(new_vocab)
    vocab=len(palmtree.vocab)
    
    new_vocab = Vocab(Counter({"<sos_p>": 1}))
    palmtree.vocab.extend(new_vocab)
    vocab=len(palmtree.vocab)
    token_seq = [1, 2, 3, 4, 5, 4683, 4684, 38]   # 例子token序列
    words = palmtree.vocab.from_seq(token_seq)
    sentence = palmtree.vocab.from_seq(token_seq, join=True)
    print(words)
    print(sentence)


    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=8)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    decoder = decoder.to(device)

    weight = palmtree.model.embedding.token.weight
    weight = weight.to(device)
    embd_dim = weight.size(1)
    new_weight = torch.cat([weight, torch.randn(1, embd_dim).to(device)]) 
    newer_weight = torch.cat([new_weight, torch.randn(1, embd_dim).to(device)])
    palmtree.model.embedding.token.weight = nn.Parameter(newer_weight)
    print(palmtree.model.embedding.token.weight.shape)
    
    model=Seq2Seq(encoder=palmtree.model,decoder=decoder,vocab_size=vocab,
                  beam_size=10,max_length=200,
                  sos_id=4684,eos_id=4683)
    
    if args.load_model_path is not None:
        print("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.do_train:
        examples = read_examples('/home/kingdom/桌面/dataset/test1/')
        train_examples = examples
        train_features = convert_examples_to_features(train_examples)
        max_length = 10
        min_length = 2
        # 过滤掉长度超过 max_length 的特征
        filtered_features = [
            f for f in train_features if len(f.source_ids) <= max_length
            and len(f.source_ids) >= min_length
            and len(f.source_mask) <= max_length
            and len(f.source_mask) >= min_length   
            and len(f.target_ids) <= max_length
            and len(f.target_ids) >= min_length
            and len(f.target_mask) <= max_length 
            and len(f.target_mask) >= min_length
            ]
        print("原始特征数量:", len(train_features))
        print("过滤后的特征数量:", len(filtered_features))
        train_features = filtered_features
        # 填充三维特征
        padded_features = []
        for f in train_features:
            
            padded_feature = {
                'source_ids': torch.cat([f.source_ids, torch.zeros(max_length - f.source_ids.shape[0], f.source_ids.shape[1], dtype=torch.long)]),
                'source_mask': torch.cat([f.source_mask, torch.zeros(max_length - f.source_mask.shape[0], f.source_mask.shape[1], dtype=torch.long)]),
                'target_ids': torch.cat([f.target_ids, torch.zeros(max_length - f.target_ids.shape[0], f.target_ids.shape[1], dtype=torch.long)]),
                'target_mask': torch.cat([f.target_mask, torch.zeros(max_length - f.target_mask.shape[0], f.target_mask.shape[1], dtype=torch.long)])
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
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
        num_train_optimization_steps = 100000
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-5},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*32*0.1,
                                                    num_training_steps=len(train_dataloader)*32)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", 32)
        print("  Num epoch = %d", num_train_optimization_steps*32//len(train_examples))
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        # bar = range(num_train_optimization_steps)
        bar = range(150000)
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
            if (global_step + 1)%100==0:
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
            if args.do_eval and ((global_step + 1) %5000 == 0) and eval_flag:
                
                # preds = model(source_ids=source_ids.long(),source_mask=source_mask.long())  
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_example('/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/addr2line-gcc.csv')
                    eval_features = convert_examples_to_features(eval_examples)
                    eval_filtered_features = [
                        f for f in eval_features if len(f.source_ids) <= max_length and
                        len(f.source_ids) >= min_length
                        and len(f.source_mask) <= max_length
                        and len(f.source_mask) >= min_length   
                        and len(f.target_ids) <= max_length
                        and len(f.target_ids) >= min_length
                        and len(f.target_mask) <= max_length 
                        and len(f.target_mask) >= min_length
                        ]
                    print("原始eval特征数量:", len(eval_features))
                    print("过滤后的eval特征数量:", len(eval_filtered_features))
                    eval_features = eval_filtered_features
                    # 填充三维特征
                    eval_padded_features = []
                    for f in eval_features:
                        eval_padded_feature = {
                        'source_ids': torch.cat([f.source_ids, torch.zeros(max_length - f.source_ids.shape[0], f.source_ids.shape[1])]),
                        'source_mask': torch.cat([f.source_mask, torch.zeros(max_length - f.source_mask.shape[0], f.source_mask.shape[1])]),
                        'target_ids': torch.cat([f.target_ids, torch.zeros(max_length - f.target_ids.shape[0], f.target_ids.shape[1])]),
                        'target_mask': torch.cat([f.target_mask, torch.zeros(max_length - f.target_mask.shape[0], f.target_mask.shape[1])])
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
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)
                
                print("\n***** Running evaluation *****")
                print("  Num examples = %d", len(eval_examples))
                print("  Batch size = %d", 32)
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
                output_file_path = '/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result_eval.csv'
                if os.path.exists(output_file_path):
                    os.remove(output_file_path)
                    print("DELETE")
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_example("/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/addr2line-gcc.csv")
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples)
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
                    print("原始eval特征数量:", len(eval_features))
                    print("过滤后的eval特征数量:", len(eval_filtered_features))    
                    eval_features = eval_filtered_features
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
                    all_source_ids = torch.stack([f['source_ids'] for f in eval_features])
                    all_source_mask = torch.stack([f['source_mask'] for f in eval_features])
                    all_target_ids = torch.stack([f['target_ids'] for f in eval_features])
                    all_target_mask = torch.stack([f['target_mask'] for f in eval_features]) 
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                            # Calculate bleu
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)

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
                        df.to_csv('/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result_eval.csv', index=False)

                model.train()
                dev_bleu=round(_bleu('/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result_eval.csv'))
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
        files=[]
        files.append("/home/kingdom/桌面/dataset/7ztest/call-1.csv")
        # if args.test_filename is not None:
        #     files.append("/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/ar-clang.csv")
        for idx,file in enumerate(files):   
            print("Test file: {}".format(file))
            eval_examples = read_example(file)
            eval_features = convert_examples_to_features(eval_examples)
            max_length = 10
            min_length = 1
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
            print("原始eval特征数量:", len(eval_features))
            print("过滤后的eval特征数量:", len(eval_filtered_features))    
            eval_features = eval_filtered_features
            # 填充三维特征
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

            all_source_ids = torch.stack([f['source_ids'] for f in eval_features])
            all_source_mask = torch.stack([f['source_mask'] for f in eval_features])
            all_target_ids = torch.stack([f['target_ids'] for f in eval_features])
            all_target_mask = torch.stack([f['target_mask'] for f in eval_features]) 
            eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)

            model.eval() 
            p=[]
            rows = []
            output_file_path = '/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result_test.csv'
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
                print("DELETE")
            # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch                    
                with torch.no_grad():
                    print("source_ids",source_ids)
                    print("source_mask",source_mask)
                    preds = model(source_ids=source_ids,source_mask=source_mask)
                    print("preds.shape",preds.shape)
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
                        print(source)
                        print(text)
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
                        rows.append({'source': source, 'target': target, 'text': text})
                        p.append(text)
                    df = pd.DataFrame(rows)
                    df.to_csv('/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result_test.csv', index=False)

            model.train()
            # dev_bleu=round(_bleu('/home/kingdom/PalmTree-master/code2code/code-to-code-trans/code/result3.csv'))
            # print("  %s = %s "%("bleu-4",str(dev_bleu)))
            # predictions=[]
            # accs=[]
            
            # with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
            #     for ref,gold in zip(p,eval_examples):
            #         predictions.append(str(gold.idx)+'\t'+ref)
            #         f.write(ref+'\n')
            #         f1.write(gold.target+'\n')    
            #         accs.append(ref==gold.target)
            # # dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file), 
            # #                      os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)
            # # logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            # # logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            # # logger.info("  "+"*"*20)

if __name__ == "__main__":
    main()
    