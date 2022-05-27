# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""runner of BERTbaseline."""

from __future__ import absolute_import, division, print_function
import datetime
import argparse
import csv
import logging
import os
import random
import sys
import pickle
import numpy as np
import torch
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import glob

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (BertConfig, BertForMultipleChoice, BertTokenizer,
                            ElectraConfig, ElectraTokenizer, RobertaConfig, RobertaTokenizer, RobertaForMultipleChoice)
# from modeling import (ElectraForMultipleChoicePlus, Baseline, BertBaseline, RobertaBaseline, BertForMultipleChoicePlus, RobertaForMultipleChoicePlus)
# from model.bertbaseline import aBert
# from bert_lstm import Bert_LSTM
# from model.bert_lstm_graph import Bert_LSTM_graph
# from bert_lstm_attention_orig import Bert_LSTM_attention_1,Bert_LSTM_attention_2
# from bert_v1 import Bert_v1
from model.bert_v2 import Bert_v2
from model.bert_v3 import Bert_v3
# from model.bert_graph import Bert_graph
# from model.bert_att_syn import Bert_att_syn
from model.bert_v4 import Bert_v4
# from model.bert_v5 import Bert_v5
from model.bert_v6 import Bert_v6
from model.bert_v7 import Bert_v7
# # from model.bert_v8 import Bert_v8
# from model.bert_lstm_attention import Bert_attention_3
# from model.bert_lstm_attention_orig import Bert_LSTM_attention_1
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
import re
import os

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # 'bert_baseline': (BertConfig, aBert, BertTokenizer),
    # 'bert_lstm': (BertConfig, Bert_LSTM, BertTokenizer),
    # 'bert_lstm_graph':(BertConfig, Bert_LSTM_graph, BertTokenizer),
    # # 'bert_graph':(BertConfig, Bert_graph, BertTokenizer),
    # 'bert_attention_1':(BertConfig, Bert_LSTM_attention_1, BertTokenizer),
    # # 'bert_attention_2':(BertConfig, Bert_LSTM_attention_2, BertTokenizer),
    # 'bert_attention_3':(BertConfig, Bert_attention_3, BertTokenizer),
    # 'bert_v1':(BertConfig, Bert_v1, BertTokenizer),
    'bert_v2':(BertConfig, Bert_v2, BertTokenizer),
    'bert_v4':(BertConfig, Bert_v4, BertTokenizer),
    # 'bert_v5':(BertConfig, Bert_v5, BertTokenizer),
    'bert_v6':(BertConfig, Bert_v6, BertTokenizer),
    'bert_v7':(BertConfig, Bert_v7, BertTokenizer),
    # 'bert_v8':(BertConfig, Bert_v8, BertTokenizer),
    'bert_v3':(BertConfig, Bert_v3, BertTokenizer),
    # 'bert_att_syn':(BertConfig, Bert_att_syn, BertTokenizer),
    # 'roberta': (RobertaConfig, RobertaForMultipleChoicePlus, RobertaTokenizer),
    # 'electra': (ElectraConfig, ElectraForMultipleChoicePlus, ElectraTokenizer)
}

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, adj_matrix_speaker=None, adj_matrix_mention=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.adj_matrix_speaker = adj_matrix_speaker
        self.adj_matrix_mention = adj_matrix_mention


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, choices_features, label, adj_matrix_speaker, adj_matrix_mention):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'sep_pos': sep_pos,
                'turn_ids': turn_ids
            }
            for input_ids, input_mask, segment_ids, sep_pos, turn_ids in choices_features
        ]
        self.label = label
        self.adj_matrix_speaker = adj_matrix_speaker
        self.adj_matrix_mention = adj_matrix_mention

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    # def get_train_examples(self, data_dir):
    #     """Gets a collection of `InputExample`s for the train set."""
    #     raise NotImplementedError()

    # def get_dev_examples(self, data_dir):
    #     """Gets a collection of `InputExample`s for the dev set."""
    #     raise NotImplementedError()
    def get_examples(self, data_dir, mode):
        """Gets a collection of `InputExample`s for the train/dev/test set.""" 
        raise NotImplementedError()

    def get_labels(self, max_previous_utterance):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class IRCProcessor(DataProcessor):
    """Processor for IRC dataset"""
    def get_examples(self, data_dir, mode, max_previous_utterance):
        logger.info("LOOKING AT {} {}".format(data_dir, mode))
        data_dir = os.path.join(data_dir, mode)
        data_dir = data_dir+'/'
        return self._reshape_examples(self._get_examples(data_dir,mode), mode, max_previous_utterance)
    
    # def get_dev_examples(self, data_dir, max_previous_utterance):
    #     logger.info("LOOKING AT {} dev".format(data_dir))
    #     data_dir = os.path.join(data_dir, 'dev')
    #     return self.reshape_examples(self.get_examples(data_dir,'dev'), 'dev', args.max_previous_utterance)
        
    # def get_test_examples(self, data_dir, max_previous_utterance):
    #     logger.info("LOOKING AT {} test".format(data_dir))
    #     data_dir = os.path.join(data_dir, 'test')
    #     return self.reshape_examples(self.get_examples(data_dir,'test'), 'test', args.max_previous_utterance)

    def _get_examples(self, data_dir, mode):
        logger.info("LOOKING AT {}, {}".format(data_dir, mode))
        suffix = [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]
        examples = []
        files = os.listdir(data_dir)
        ascii_files = [data_dir+f for f in files if f.endswith(suffix[1])]
        for file in ascii_files:
            example = {}
            filename = file.split('/')[-1]
            filename = filename[0:10].split('-')
            filename = [int(fn) for fn in filename]
            example['filename'] = filename
            # print("filename:", filename)
            # print("filename:", example['filename'])
            text_ascii = [l.strip().split() for l in open(file)] #all utters in one recording
            #print('text_ascii:',text_ascii[0])
            example['text_ascii'] = text_ascii
            #read labels
            filename_ = file[:-len(suffix[1])]
            labels = [l.strip().split() for l in open(filename_+suffix[0])]
            #print('labels:',labels[0])
            example['labels'] = labels
            examples.append(example)
        print('examples:', examples[0]['filename'])
        return examples
    
    def get_labels(self, max_previous_utterance):
        return [i for i in range(max_previous_utterance)]

    def _reshape_examples(self, examples, mode, max_previous_utterance):
        """turn the examples into trainable form"""
        # print('len examples:',len(examples))
        filenames = []
        reshaped_examples = []
        for example in examples:
            # print("example[0]:",example)
            filenames.append(example['filename'])
            the_main_text = [int(i[1]) for i in example['labels']]
            # print('filename:',example['filename'])#,' the_main_text:', the_main_text)
            begin = min(the_main_text)
            end = max(the_main_text)
            # print("from {} to {}".format(int(begin), int(end)))
            for i in range(int(begin), int(end)+1):
                # reshaped_example['id'] = i
                # id = "%s-%s" % (example['filename'], i)
                id = [0,0,0,0]
                id[0:3] = example['filename'][0:3]
                id[3] = i
                # print("filename:", example['filename'])
                # print('id:',id)
                # guids = []
                # considered_pairs = []
                text_a = []
                text_b = []
                # if mode == 'train':
                label = []
                for l in example['labels']:
                    if int(l[1]) == i:
                        label.append(int(l[0]))
                # print(label)
                label = i-max(label) #use the nearest one, label is the distance
                if label>=max_previous_utterance:
                    label=0
                # else:
                #     label = None
                for j in range(0, max_previous_utterance):
                    # pair = [example['text_ascii'][i-j], example['text_ascii'][i]]
                    text_b.append(example['text_ascii'][i-j])
                    text_a.append(example['text_ascii'][i])
                    # considered_pairs.append(pair)
                    # guids.append((i,j))
                adj_matrix_speaker = [[0]*max_previous_utterance]*max_previous_utterance
                adj_matrix_mention = [[0]*max_previous_utterance]*max_previous_utterance
                all_speakers = [] #50个句子中所有的speakers
                # print("lenth: ",len(text_b))
                for (i,line) in enumerate(text_b):
                    if line[0] == '===':
                        continue
                    # print('speaker1:',line) #list
                    try:
                        speaker1 = ' '.join(line).split('<')[1].split('>')[0].lower()
                    except IndexError:
                        #no speaker
                        continue
                    # print("speaker1:",speaker1)
                    assert speaker1 is not None
                    if speaker1 not in all_speakers:
                        all_speakers.append(speaker1)
                    for (j,pre) in enumerate(text_b):
                        if pre[0] == '===':
                            continue
                        # print("pre:",pre)
                        try:
                            speaker2 = "".join(pre).split('<')[1].split('>')[0].lower()
                        except IndexError:
                            # print("pre:",pre)
                            continue
                        if speaker1 == speaker2:
                            # print("i,j:",i,j)
                            adj_matrix_speaker[i][j]=1
                            adj_matrix_speaker[j][i]=1
                # print("adj_matrix_speaker:", adj_matrix_speaker)
                for (i,line) in enumerate(text_b):
                    if line[0] == '===':
                        continue
                    speaker_mentioned = []
                    for one in all_speakers:
                        if one in ''.join(line): #line mentions one
                            if one not in speaker_mentioned:
                                speaker_mentioned.append(one)
                    # print("speaker mentioned by this line:", speaker_mentioned)
                    for (j,pre) in enumerate(text_b):
                        if pre[0] == '===':
                            continue
                        try:
                            speaker_this = "".join(pre).split('<')[1].split('>')[0].lower()
                        except IndexError:
                            continue
                        if speaker_this in speaker_mentioned:
                            adj_matrix_mention[i][j]=1
                # print('adj_matrix_mention:', adj_matrix_mention)
            
                # print(label)
                # reshaped_example['pairs'] = considered_pairs
                # print('id:',id)
                # reshaped_example['label'] = label
                reshaped_examples.append(
                    InputExample(
                        guid=id,
                        text_a=text_a,
                        text_b=text_b,
                        label=label,
                        adj_matrix_speaker=adj_matrix_speaker,
                        adj_matrix_mention=adj_matrix_mention
                    )
                )
        # if mode != 'train': 
        #     print(reshaped_examples[0].guid,reshaped_examples[0].text_a,reshaped_examples[0].text_b,reshaped_examples[0].label,len(reshaped_examples[0].text_a))
        # print('reshaped:',len(reshaped_examples))
        return reshaped_examples, filenames

def convert_examples_to_features(examples, label_list, max_seq_length, max_utterance_num,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    # print(len(examples))
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        choices_features = []
        all_tokens = []
        # if example.label>=50:
        #     print('more than 50',example.label,example.guid)

        for ending_idx, (text_a, text_b) in enumerate(zip(example.text_a, example.text_b)):
            # print('texta',text_a)
            text_a = " ".join(text_a)
            text_b = " ".join(text_b)
            # print('texta string', text_a,' textb string ', text_b)
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)
            tokens_b = tokens_b + ["[SEP]"]
            # print('tokensa', tokens_a)

            # tokens_b = []

            # for idx, text in enumerate(text_b):
            #     if len(text.strip()) > 0:
            #         tokens_b.extend(tokenizer.tokenize(text) + ["[SEP]"])
            # print('tokensb', tokens_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)

            tokens = ["[CLS]"]
            turn_ids = [0]

            context_len = []
            sep_pos = []
            
            tokens_b_raw = " ".join(tokens_b)
            tokens_b = []
            current_pos = 0
            for toks in tokens_b_raw.split("[SEP]")[-max_utterance_num - 1:-1]:
                context_len.append(len(toks.split()) + 1)
                tokens_b.extend(toks.split())
                tokens_b.extend(["[SEP]"])
                current_pos += context_len[-1]
                turn_ids += [len(sep_pos)] * context_len[-1]
                sep_pos.append(current_pos)
                
            tokens += tokens_b #cls b sep a sep

            segment_ids = [0] * (len(tokens))

            tokens_a += ["[SEP]"]
            tokens += tokens_a
            segment_ids += [1] * (len(tokens_a))
            
            turn_ids += [len(sep_pos)] * len(tokens_a) 
            sep_pos.append(len(tokens) - 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            turn_ids += padding

            context_len += [-1] * (max_utterance_num - len(context_len))
            sep_pos += [0] * (max_utterance_num + 1 - len(sep_pos))

            assert len(sep_pos) == max_utterance_num + 1
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(context_len) == max_utterance_num 
            assert len(turn_ids) == max_seq_length 

            choices_features.append((input_ids, input_mask, segment_ids, sep_pos, turn_ids))
            all_tokens.append(tokens)

        if example.label!=None:
            label_id = label_map[example.label]
        else:
            label_id = None
        

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: ", (example.guid))
            for choice_idx, (input_ids, input_mask, segment_ids, sep_pos, turn_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("tokens: %s" % " ".join([str(x) for x in all_tokens[choice_idx]]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("sep_pos: %s" % " ".join([str(x) for x in sep_pos]))
                logger.info("turn_ids: %s" % " ".join([str(x) for x in turn_ids]))
                logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(
                example_id = example.guid, 
                choices_features = choices_features,
                label = label_id,
                adj_matrix_speaker=example.adj_matrix_speaker,
                adj_matrix_mention=example.adj_matrix_mention
                )
        )

    return features
            

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop(0)

def get_p_at_n_in_m(pred, n, m, ind):
    pos_score = pred[ind]
    curr = pred[ind:ind + m]
    curr = sorted(curr, reverse=True)

    if curr[n - 1] <= pos_score:
        return 1
    return 0

def mean_average_precision(sort_data):
    # to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index + 1)
    return sum_precision / count_1


def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0


def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    
    return 1.0 * select_lable.count(1) / sort_lable.count(1)

def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5

def evaluate_douban(pred, label):
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    total_num = 0
    data = []
    #print(label)
    for i in range(0, len(label)):
        if i % 10 == 0:
            data = []
        data.append((float(pred[i]), int(label[i])))
        if i % 10 == 9:
            total_num += 1
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5
    # print('total num: %s' %total_num)
    # print('MAP: %s' %(1.0*sum_m_a_p/total_num))
    # print('MRR: %s' %(1.0*sum_m_r_r/total_num))
    # print('P@1: %s' %(1.0*sum_p_1/total_num))
    return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
            1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)

def evaluate(pred, label):
    # assert len(data) % 10 == 0

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = int(len(pred) / 10)

    for i in range(0, length):
        ind = i * 10
        assert label[ind] == 1

        p_at_1_in_2 += get_p_at_n_in_m(pred, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(pred, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(pred, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(pred, 5, 10, ind)

    return (p_at_1_in_2 / length, p_at_1_in_10 / length, p_at_2_in_10 / length, p_at_5_in_10 / length)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def ComputeR10(scores,labels,count = 10):
    total = 0
    correct1 = 0
    correct5 = 0
    correct2 = 0
    correct10 = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if np.argmax(sublist) < 1:
                correct1 = correct1 + 1
            if np.argmax(sublist) < 2:
                correct2 = correct2 + 1
            if np.argmax(sublist) < 5:
                correct5 = correct5 + 1
            if np.argmax(sublist) < 10:
                correct10 = correct10 + 1
    print(correct1, correct5, correct10, total)
    return (float(correct1)/ total, float(correct2)/ total, float(correct5)/ total, float(correct10)/ total)

def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    return (float(correct)/ total)

def Compute_R4_2(preds, labels):
    p2 = 0
    for i in range(len(preds)):
        j = sorted(list(preds[i]), reverse = True)
        if j.index(preds[i][labels[i]]) <= 1:
            p2 += 1
    return p2 / len(preds)

def Compute_MRR(preds, labels):
    mrr = 0
    for i in range(len(preds)):
        j = sorted(list(preds[i]), reverse = True)
        mrr += 1 / (j.index(preds[i][labels[i]]) + 1)
    
    return mrr / len(preds)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if preds.shape[1] == 1:
        preds_class = np.ones(preds.shape)
        preds_class[preds < 0] = 0
    else:
        preds_class = np.argmax(preds, axis=1)
    preds_logits = preds[:, 1]

    # if task_name in ["ubuntu", "ecd"]:
    #     return {"acc": simple_accuracy(preds_class, labels), "recall@10":ComputeR10(preds_logits, labels), "recall@2":ComputeR2_1(preds_logits, labels), "DAM":evaluate(preds_logits, labels)}
    # elif task_name == 'douban':
    #     return {"DAM":evaluate_douban(preds_logits, labels)}
    # elif task_name in ['mutual']:
    return {"R4_1": simple_accuracy(preds_class, labels), "R4_2": Compute_R4_2(preds, labels), "MRR:": Compute_MRR(preds, labels)}
    # else:
    #     raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='../../lw/data/',
                        type=str,
                        help="Location of data.")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--model_type", default="bert", type = str,
                        help = "Pre-trained Model selected in the list: bert, roberta, electra.")
    parser.add_argument("--task_name",
                        default="bert_baseline",
                        type=str,
                        help="version of model to train:baseline/version2/version3...")
    parser.add_argument("--dataset_name",
                        default="IRC",
                        type=str,
                        help="dataset name")
    parser.add_argument("--output_dir",
                        default="./output/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_previous_utterance",
                        default=50,
                        type=int,
                        help="The maximum of previous utterances considerated.")
    parser.add_argument("--max_utterance_num",
                        default=1,
                        type=int,
                        help="The maximum of previous utterances considerated in one pairwise input.")
    parser.add_argument("--cache_flag", #unused
                        default="v1",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--max_grad_norm",  #unused
                        default = 1.0, 
                        type = float,
                        help = "The maximum grad norm for clipping")
    parser.add_argument("--cache_dir",
                        default='./cached_models',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    # parser.add_argument("--baseline",
    #                     action='store_true',
    #                     help="Whether to run baseline.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=4e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--num_rnn",
    #                     default=1,
    #                     type=int,
    #                     help="RNN.")
    # parser.add_argument("--num_decouple",
    #                     default=1,
    #                     type=int,
    #                     help="Decoupling Layers.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', #unused?
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # processors = {
    #     "ubuntu": UbuntuProcessor,
    #     'douban': DoubanProcessor,
    #     'ecd': UbuntuProcessor,
    #     "mutual": MuTualProcessor
    # }
    processors = {'irc': IRCProcessor,
                }

    # output_modes = {
    #     "ubuntu": "classification",
    #     "mutual": "classification",
    #     'douban': "classification",
    #     'ecd': 'classification'
    # }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:    
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or  `do_test`must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    dataset_name = args.dataset_name.lower()
    print("task_name(model_version):{}, dataset_name:{}".format(task_name, dataset_name))

    if dataset_name not in processors:
        raise ValueError("Task not found: %s" % (dataset_name))

    processor = processors[dataset_name]()
    # output_mode = output_modes[task_name]

    label_list = processor.get_labels(args.max_previous_utterance)
    num_labels = len(label_list) 

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.task_name]

    # if args.baseline:
    #     if args.model_type == 'electra':
    #         model_class = Baseline
    #     elif args.model_type == 'bert':
    #         model_class = BertBaseline
    #     elif args.model_type == 'roberta':
    #         model_class = RobertaBaseline

    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name, 
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples, filenames = processor.get_examples(args.data_dir, "train", args.max_previous_utterance)
        print(len(train_examples))
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.model_name_or_path.split('/'))).pop(), "train",str(args.task_name.split('_')[0]), str(args.max_seq_length),
            str(args.max_utterance_num), str(args.cache_flag))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        # (batch_size, 1, seq_len) 这里根据processor改
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        #all_response_len = torch.tensor(select_field(train_features, 'response_len'), dtype=torch.long)
        # all_sep_pos = torch.tensor(select_field(train_features, 'sep_pos'), dtype=torch.long)
        # all_turn_ids = torch.tensor(select_field(train_features, 'turn_ids'), dtype = torch.long)

        # if output_mode == "classification":
        all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)
        all_adj_speaker = torch.tensor([f.adj_matrix_speaker for f in train_features], dtype=torch.long)
        all_adj_mention = torch.tensor([f.adj_matrix_mention for f in train_features], dtype=torch.long)
        all_guid = torch.tensor([f.example_id for f in train_features], dtype=torch.long)
        # elif output_mode == "regression":
        #     all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float)
        # 这里要解决label是list的问题
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_adj_speaker, all_adj_mention, all_guid)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            eval_examples,filenames = processor.get_examples(args.data_dir, "dev", args.max_previous_utterance)
            cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.model_name_or_path.split('/'))).pop(), "eval",str(args.task_name.split('_')[0]), str(args.max_seq_length),
            str(args.max_utterance_num), str(args.cache_flag))
        elif args.do_test:
            eval_examples, filenames = processor.get_examples(args.data_dir, "test", args.max_previous_utterance)
            cached_train_features_file = args.data_dir + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.model_name_or_path.split('/'))).pop(), "test",str(args.task_name.split('_')[0]), str(args.max_seq_length),
            str(args.max_utterance_num), str(args.cache_flag))
        eval_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving eval/test features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        # all_sep_pos = torch.tensor(select_field(eval_features, 'sep_pos'), dtype=torch.long)
        # all_turn_ids = torch.tensor(select_field(eval_features, 'turn_ids'), dtype = torch.long)
        # # 这里要解决label是list的问题
        # if output_mode == "classification":
        all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_adj_speaker = torch.tensor([f.adj_matrix_speaker for f in eval_features], dtype=torch.long)
        all_adj_mention = torch.tensor([f.adj_matrix_mention for f in eval_features], dtype=torch.long)
        all_guid = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)
        # elif output_mode == "regression":
        #     all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.float)
        print("eval size:",all_input_ids.size(),all_input_mask.size(),all_segment_ids.size(),all_label_ids.size(),all_adj_speaker.size(),all_adj_mention.size())
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_adj_speaker, all_adj_mention, all_guid)
        # Run prediction for full data

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            #nb_tr_examples = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                # print('batch6:',batch[6].size(),type(batch[6]),batch[6][0])
                # print('batch3:',batch[3].size(),type(batch[3]),batch[3][0])
                # guid = batch[6].detach().cpu().numpy()
                # guid = np.append(guid, batch[6].detach().cpu().numpy(), axis=0)
                # print('guid:',guid,type(guid))
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                
                          'labels': batch[3],
                          'adj_matrix_speaker':batch[4],
                          'adj_matrix_mention':batch[5]}
                
                #input_ids, input_mask, segment_ids, response_len, sep_pos, label_ids = batch

                output = model(**inputs)
                loss = output[0]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.detach().item()
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1

            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(args.output_dir, str(epoch) + "_" + WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = None

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                # print('batch6:',batch[6].size(),type(batch[6]),batch[6][0])
                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                        #   'sep_pos': batch[3],
                        #   'turn_ids': batch[4],
                          'labels': batch[3],
                          'adj_matrix_speaker':batch[4],
                          'adj_matrix_mention': batch[5]}
                    #outputs = eval_model(**inputs)
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.detach().mean().item()
                
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                    guid = batch[6].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                    guid = np.append(guid, batch[6].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps

            if preds.shape[1] == 1:
                preds_class = np.ones(preds.shape)
                preds_class[preds < 0] = 0
            else:
                preds_class = np.argmax(preds, axis=1)

            f = open(os.path.join(args.output_dir, 'epoch_{}_eval_results.txt'.format(epoch)), 'w')
            for k in range(preds.shape[0]):
                f.write(str(guid[k][0])+'-'+str(guid[k][1]) +'-'+str(guid[k][2])+ ':' +str(guid[k][3]) + ' '+ str(guid[k][3]-preds_class[k]) + ' ' + str(guid[k][3]-out_label_ids[k]) + ' ' + '-' + '\n')
            f.close()
            result = compute_metrics(task_name, preds, out_label_ids)#改metrics
            loss = tr_loss / nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
