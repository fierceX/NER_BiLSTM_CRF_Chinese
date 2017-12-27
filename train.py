import os
import re
import pandas as pd
import mxnet as mx
from mxnet import autograd as ag, ndarray as nd, gluon
from mxnet.gluon import Block, nn, rnn
import mxnet.optimizer as optim
import sys
from model import *
from tqdm import tqdm
import pickle

dirs = './data/'
files = [ files for root,dirss,files in os.walk(dirs)][0]

tags = []
for zz in files:
    if zz.find('txtoriginal') > 0:
        bb = os.path.join(dirs,zz.replace('.txtoriginal',''))
        try:
            p = pd.read_csv(bb,header=-1, delimiter="\t")
            for i in range(p.shape[0]):
                tags.append(p.iloc[i][3])
        except:
            pass

idx2tag = list(set(tags))
btag2idx = dict([(char,i) for i,char in enumerate(idx2tag)])
idx2btag = { 0:"B",  1:"I", 2:"O",3:"S",4:"X"}

train_data = []
for zz in files:
    if zz.find('txtoriginal') > 0:
        bb = os.path.join(dirs,zz.replace('.txtoriginal',''))
        f = open(os.path.join(dirs,zz),encoding="utf-8")
        ff = f.read()
        ff = ff.replace("\n", "")
        f.close()
        bio = ['X' for i in range(len(ff))]
        try:
            p = pd.read_csv(bb,header=-1, delimiter="\t")
            for i in range(p.shape[0]):
                s = p.iloc[i][1]
                e = p.iloc[i][2]
                for j in range(s,e+1):
                    bio[j] = idx2btag[btag2idx[p.iloc[i][3]]]
        
        except:
            pass
        train_data.append((list(ff),bio))

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

word2idx = {}
for sentence, tags in train_data:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

tag2idx = {"B": 0, "I": 1, "O": 2, "S":3,"X":4, START_TAG: 5, STOP_TAG: 6}

model = BiLSTM_CRF(len(word2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM,ctx=mx.gpu())
model.initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.gpu())
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})

for epoch in range(2):
    for sentence, tags in tqdm(train_data):
        with ag.record():
            sentence_in = prepare_sequence(sentence, word2idx)
            targets = nd.array([tag2idx[t] for t in tags])

            neg_log_likelihood = model.neg_log_likelihood(sentence_in.as_in_context(mx.gpu()), targets.as_in_context(mx.gpu()))

            neg_log_likelihood.backward()
        optimizer.step(1)

model.save_params('model.params')
f = open('wordtagidx','wb')
wordtagidx = [word2idx,tag2idx,idx2tag]
pickle.dump(wordtagidx,f)
f.close()