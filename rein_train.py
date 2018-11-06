from __future__ import division, unicode_literals
import argparse

import time
import math
import random
import torch.nn as nn, torch
import torch.nn.init as init
import torch.optim as optim
import os
import numpy as np
import pickle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

from onmt.utils.logging import init_logger
from onmt.translate.translator_new import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts


class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = 1
        self.drop = nn.Dropout(0.4)
        self.direction = 2 # 22
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size, num_layers=self.num_lyr, bidirectional=True, batch_first=True, dropout=0.4)

    def forward(self, inp):
        x = inp.view(-1, inp.size(2))
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)

        bt_siz, seq_len = x_emb.size(0), x_emb.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2 * i:2 * i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)

        x_hid = x_hid[self.num_lyr - 1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)

        return x_hid


class Policy_Network(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, batch_size, translator):
        super(Policy_Network, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = BaseEncoder(vocab_size, 300, 400).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=0.2)

        self.total_reward = 0
        self.num_reward = 0
        self.total_batch = 0

        self.hidden = self.init_hidden(batch_size)

        self.translator = translator

    def init_hidden(self, batch_size):
        return(torch.randn(1, batch_size, self.hidden_dim).cuda(), torch.randn(1, batch_size, self.hidden_dim).cuda())

    def baseline_score(self, reward, num_reward):
        return reward / num_reward

    def calculate_reward(self, list, pred):
        max = 0
        for line in list:
            cos = cos_sim(translator._translate_batch(line), pred)
            if cos > max:
                max = cos
        return max

    def forward(self, input):
        input = self.embedding(input)
        input = input.transpose(0, 1)

        outputs, self.hidden = self.lstm(input, self.hidden)

        output = self.dropout_layer(self.hidden[0][-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        pred_index = (output.max(1)[1])

        # Calculate reward & base_score
        reward_list = []
        base_list = []

        for batch_index in range(len(pred_index)):
            response_file = open("distill_files/Response.txt{}".format(pred_index[batch_index]), 'r')
            predict_output = open("distill_files/train_output_{}.txt".format(pred_index[batch_index]), 'r')
            reward = calculate_reward(response_file.readlines(), predict_output.readlines()[total_batch])
            total_batch += 1
            predict_output.close()
            response_file.close()

            # addtional line
            reward = 1 - reward
            reward_list.append(reward)
            self.total_reward += reward
            self.num_reward += 1
            base_list.append(self.baseline_score(self.total_reward, self.num_reward))

        reward_list = torch.Tensor(reward_list).cuda()
        base_list = torch.Tensor(base_list).cuda()

        new_output = output.transpose(0, 1) * (reward_list - base_list)
        new_output = new_output.transpose(0, 1)

        return new_output, output

def cos_sim(list1, list2):
    return nn.functional.cosine_similarity(list1, list2)

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)

##############################################################################################################################################

def RL_train_model(RL_model, optimizer, dataloader, num_epochs, inv_dict):
    criterion = nn.NLLLoss()
    if use_cuda:
        criterion.cuda()

    RL_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        start_index = 0
        temp_loss = 10000
        for i_batch, sample_batch in enumerate(dataloader):
            temp_list = rele_list[start_index : start_index + len(sample_batch)]
            start_index += len(sample_batch)

            RL_model.zero_grad()
            RL_model.hidden = RL_model.init_hidden(len(sample_batch))

            pred_rele, pred_base = RL_model(sample_batch)

            pred_base = pred_base.max(1)[1]
            loss = criterion(pred_rele, pred_base)

            if temp_loss > loss:
                loss.backward()
            temp_loss = loss.item()
            optimizer.step()
            total_loss += loss.item()

        #print("Epoch : {} / Train loss: {}".format(epoch, total_loss))

        if (epoch + 1) % 10 == 0:
            model_fname = './save/new_RL_model_epoch{}.pt'.format(epoch)
            torch.save(RL_model.state_dict(), model_fname)

##############################################################################################################################################


def main():
    with open('./distill_files/w2i', 'rb') as f:
        inv_dict = pickle.load(f)

    # parameter
    N = 4
    folder_path = "distill_files/"

    f = open(folder_path + "src-train.0", 'rb')
    line_list = pickle.load(f)
    f.close()

    new_list = []
    for line in line_list:
        new_list.append(Variable(torch.LongTensor([line])).cuda())

    dataloader = DataLoader(new_list, 64, shuffle=False)

    parser = argparse.ArgumentParser(
        description='rein_train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()


    translator = build_translator(opt, report_score=True)

    RL_model = Policy_Network(len(inv_dict), 400, 128, N, 64, translator).cuda()
    optimizer = optim.SGD(RL_model.parameters(), lr=0.1, weight_decay=1e-4)

    num_epochs = 300
    RL_train_model(RL_model, optimizer, dataloader, num_epochs, inv_dict)

main()
