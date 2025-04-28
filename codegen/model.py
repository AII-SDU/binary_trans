# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

"""
序列到序列模型定义文件
主要实现了基于Transformer架构的序列到序列模型，用于二进制代码转换
"""
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
class Seq2Seq(nn.Module):
    """
    序列到序列模型，用于实现二进制代码转换
    
    参数:
        encoder: 编码器模型，例如 PalmTree
        decoder: 解码器模型，例如 Transformer decoder
        vocab_size: 词汇表大小
        beam_size: beam search的宽度
        max_length: beam search的最大目标长度
        sos_id: 目标序列的起始标记ID
        eos_id: 目标序列的结束标记ID
    """
    def __init__(self, encoder,decoder,vocab_size,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(128, 128)
        self.lm_head = nn.Linear(128, vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        # self.mha = MultiHeadAttention(n_heads=8, d_model=512)
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """
        绑定或克隆模块权重
        
        参数:
            first_module: 第一个模块
            second_module: 第二个模块
        """
        first_module.weight = second_module.weight
        print("weight shape:", first_module.weight.shape)
                  
    def tie_weights(self):
        """
        确保共享输入和输出嵌入权重
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embedding.token)        
    def get_embedding(self, source_ids):
        """
        获取输入的嵌入表示
        
        参数:
            source_ids: 输入序列的ID
            
        返回:
            嵌入表示
        """
        if len(source_ids.shape) == 3:
            flat_ids = source_ids.view(source_ids.shape[0], -1)
        else:
            flat_ids = source_ids
        token_weight = self.encoder.embedding.token.weight
        token_weight[0].data.fill_(0.0)
        embeds = token_weight[flat_ids]
        return embeds
    
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
        """
        模型前向计算
        
        参数:
            source_ids: 源序列ID，形状为[batch_size, seqs_per_batch, seq_len]
            source_mask: 源序列掩码
            target_ids: 目标序列ID
            target_mask: 目标序列掩码
            args: 其他参数
            
        返回:
            训练时: (loss, loss*active_loss.sum(), active_loss.sum())
            推理时: 预测结果
        """
        # 获取源序列的形状
        batch_size, seqs_per_batch, seq_len = source_ids.shape
        token_embedding = self.encoder.embedding.token
        # self.get_embedding(source_ids)
        outputs = []
        masks =[]
        source_outputs = []
        target_outputs = []
        target_masks =[]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # target_outputs = []
        for i in range(batch_size):
            input_ids = source_ids[i]
            if source_mask is not None:
                input_mask = source_mask[i]
            else:
                input_mask = None

            # 获取encoder的输出
            valid_output = []
            valid_mask = []
            valid_source_ids = []
            encoder_output = self.encoder.forward(input_ids, input_mask)
            for i, t in enumerate(encoder_output):
                #得到有效长度
                valid_len = input_mask[i].sum()
                #取出encoder output的有效位数
                valid_out = t[:valid_len]
                #一个encoder output batchsize（10）的循环放入有效位数
                valid_output.append(valid_out)
                #有效长度mask output batchsize的累加
                valid_mask.append(input_mask[i][:valid_len])
                valid_source_ids.append(input_ids[i][:valid_len])
            #有效长度torch累加
            combined = torch.cat(valid_output) 
            combined_mask = torch.cat(valid_mask)
            combined_source_ids = torch.cat(valid_source_ids)
            combined = combined.to(device) 
            combined_mask = combined_mask.to(device) 
            combined_source_ids = combined_source_ids.to(device) 
            #pad到200
            pad = torch.zeros(200-combined.shape[0], 128)
            pad_mask = torch.zeros(200-combined_mask.shape[0]) 
            pad_source_ids = torch.zeros(200-combined_source_ids.shape[0]) 

            pad = pad.to(device)
            pad_mask = pad_mask.to(device)
            pad_source_ids = pad_source_ids.to(device)

            padded_output = torch.cat([combined, pad], dim=0)
            padded_mask = torch.cat([combined_mask, pad_mask], dim=0)
            padded_source_ids = torch.cat([combined_source_ids, pad_source_ids], dim=0)

            # encoder_output = encoder_output.permute([1,0,2]).contiguous()
            outputs.append(padded_output)
            masks.append(padded_mask)
            source_outputs.append(padded_source_ids)

        outputs = torch.stack(outputs, dim=0)
        masks = torch.stack(masks, dim=0)
        source_outputs = torch.stack(source_outputs, dim=0)

        masks = masks.long()
        source_outputs = source_outputs.long()
        outputs = outputs.to(device) 
        masks = masks.to(device) 
        source_outputs = source_outputs.to(device) 
        outputs = outputs.permute([1,0,2]).contiguous()

        # outputs = self.encoder(source_ids, attention_mask=source_mask)
        # encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            
            # print('target_ids is not None, doing train')
            batch_size, seqs_per_batch, seq_len = target_ids.shape
            for i in range(batch_size):
                valid_output = []
                valid_mask = []  
                input_ids = target_ids[i]
                if source_mask is not None:
                    input_mask = target_mask[i]
                else:
                    input_mask = None
                for n in range(seqs_per_batch):
                    valid_len = input_mask[n].sum()
                    valid_out = input_ids[n][:valid_len]
                    valid_output.append(valid_out)
                    valid_mask.append(input_mask[n][:valid_len])

                combined = torch.cat(valid_output) 
                combined_mask = torch.cat(valid_mask)
                combined = combined.to(device) 
                combined_mask = combined_mask.to(device) 

                #pad到200
                pad = torch.zeros(200-combined.shape[0])
                pad_mask = torch.zeros(200-combined_mask.shape[0]) 

                pad = pad.to(device)
                pad_mask = pad_mask.to(device)
                padded_output = torch.cat([combined, pad], dim=0)
                padded_mask = torch.cat([combined_mask, pad_mask], dim=0)


                target_outputs.append(padded_output)
                target_masks.append(padded_mask)

            target_outputs = torch.stack(target_outputs, dim=0)
            target_masks = torch.stack(target_masks, dim=0)
            target_outputs = target_outputs.long()
            target_masks = target_masks.long()
            
            target_outputs = target_outputs.to(device) 
            target_masks = target_masks.to(device) 
            attn_target_ids = target_outputs
            source_mask = masks
            attn_mask=-1e4 *(1-self.bias[:attn_target_ids.shape[1],:attn_target_ids.shape[1]])
            # for i in range(batch_size):
            #     input_ids = target_ids[i]
            #     if target_mask is not None:
            #         input_mask = target_mask[i]
            #     else:
            #         input_mask = None
            #     # 获取encoder的输出
            #     target_encoder_output = self.encoder.forward(input_ids, input_mask)
            #     target_encoder_output = target_encoder_output.view(-1, encoder_output.size(-1))
            #     # encoder_output = encoder_output.permute([1,0,2]).contiguous()
            #     target_outputs.append(target_encoder_output)
            # target_outputs = torch.stack(target_outputs, dim=0)
            # target_outputs = target_outputs.permute([1,0,2]).contiguous()
            # print(target_outputs.shape)
            # print(tgt_embeddings.shape)
            # print(tgt_embeddings[0])
            # tgt_embeddingstgt_embeddings=self.get_embedding(target_ids)
            tgt_embeddings = self.get_embedding(attn_target_ids.long()).permute([1,0,2]).contiguous()
            # print(tgt_embeddingstgt_embeddings[0])
            # device = torch.device("cuda:0")
            # outputs = outputs.to(device) 
            # tgt_embeddings = tgt_embeddings.to(device)
            # attn_mask      = attn_mask.to(device)
            # source_mask    = source_mask.to(device)
            # self.dense     = self.dense.to(device)
            out = self.decoder(tgt_embeddings,outputs,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            target_mask = target_masks
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = attn_target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        
        else:
            #Predict 
            # print("进入preds")
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)
            source_mask = masks
            source_ids = source_outputs
            # print("source_outputs",source_outputs.shape) 
            for i in range(source_ids.shape[0]):
                # print("进入preds", i)
                context=outputs[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    if _ == 199:
                        print("beam失败")
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.get_embedding(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)          
            return preds   
        
        
class Beam(object):
    """
    Beam Search实现
    
    参数:
        size: beam宽度
        sos: 序列开始标记ID
        eos: 序列结束标记ID
    """
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
# class Beam(object):
#     def __init__(self, size,sos,eos):
#         self.size = size
#         self.tt = torch.cuda
#         # The score for each translation on the beam.
#         self.scores = self.tt.FloatTensor(size).zero_()
#         # The backpointers at each time-step.
#         self.prevKs = []
#         # The outputs at each time-step.
#         self.nextYs = [self.tt.LongTensor(size)
#                        .fill_(0)]
#         self.nextYs[0][0] = sos
#         # Has EOS topped the beam yet.
#         self._eos = eos
#         self.eosTop = False
#         # Time and k pair for finished.
#         self.finished = []

#     def getCurrentState(self):
#         "Get the outputs for the current timestep."
#         batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
#         return batch

#     def getCurrentOrigin(self):
#         "Get the backpointers for the current timestep."
#         return self.prevKs[-1]

#     def advance(self, wordLk):
#         """
#         Given prob over words for every last beam `wordLk` and attention
#         `attnOut`: Compute and update the beam search.

#         Parameters:

#         * `wordLk`- probs of advancing from the last step (K x words)
#         * `attnOut`- attention at the last step

#         Returns: True if beam search is complete.
#         """
#         numWords = wordLk.size(1)

#         # Sum the previous scores.
#         if len(self.prevKs) > 0:
#             beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

#             # Don't let EOS have children.
#             for i in range(self.nextYs[-1].size(0)):
#                 # print(f"self.nextYs[-1][{i}]",self.nextYs[-1][i].item())
#                 # print(f"self.nextYs[-2][self.prevKs[-1][{i}].item()]",self.nextYs[-2][self.prevKs[-1][i].item()].item())
#                 if len(self.nextYs) < 2 or self.nextYs[-1][i] != self._eos or self.nextYs[-2][self.prevKs[-1][i].item()] != self._eos:
#                     beamLk[i] = -1e20
#         else:
#             beamLk = wordLk[0]
#         flatBeamLk = beamLk.view(-1)
#         bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

#         self.scores = bestScores

#         # bestScoresId is flattened beam x word array, so calculate which
#         # word and beam each score came from
#         prevK = bestScoresId // numWords
#         self.prevKs.append(prevK)
#         self.nextYs.append((bestScoresId - prevK * numWords))


#         for i in range(self.nextYs[-1].size(0)):
#             if self.nextYs[-1][i] == self._eos and (len(self.nextYs) == 1 or self.nextYs[-2][self.prevKs[-1][i]] == self._eos):
#                 s = self.scores[i]
#                 self.finished.append((s, len(self.nextYs) - 1, i))

#         # End condition is when top-of-beam is EOS and no global score.
#         if len(self.nextYs) < 2 or self.nextYs[-1][i] != self._eos or self.nextYs[-2][self.prevKs[-1][i].item()] != self._eos:
#             self.eosTop = True

#     def done(self):
#         return self.eosTop and len(self.finished) >=self.size

#     def getFinal(self):
#         if len(self.finished) == 0:
#             self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
#         self.finished.sort(key=lambda a: -a[0])
#         if len(self.finished) != self.size:
#             unfinished=[]
#             for i in range(self.nextYs[-1].size(0)):
#                 if len(self.nextYs) < 2 or self.nextYs[-1][i] != self._eos or self.nextYs[-2][self.prevKs[-1][i].item()] != self._eos:
#                     s = self.scores[i]
#                     unfinished.append((s, len(self.nextYs) - 1, i)) 
#             unfinished.sort(key=lambda a: -a[0])
#             self.finished+=unfinished[:self.size-len(self.finished)]
#         return self.finished[:self.size]

#     def getHyp(self, beam_res):
#         """
#         Walk back to construct the full hypothesis.
#         """
#         hyps=[]
#         for _,timestep, k in beam_res:
#             hyp = []
#             for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
#                 hyp.append(self.nextYs[j+1][k])
#                 k = self.prevKs[j][k]
#             hyps.append(hyp[::-1])
#         return hyps
    
#     def buildTargetTokens(self, preds):
#         sentence = []
#         for pred in preds:
#             tokens = []
#             eos_count = 0
#             for tok in pred:
#                 if tok == self._eos:
#                     eos_count += 1
#                     if eos_count >= 2:
#                         break
#                 else:
#                     eos_count = 0
#                 tokens.append(tok)
#             sentence.append(tokens)
#         return sentence
        
