from model import *
import sys
import pickle

f = open('wordtagidx','rb')
wordtagidx = pickle.load(f)


class NER():
    def __init__(self,modelpath,wordtagidx,ctx=mx.cpu()):
        word2idx,tag2idx,idx2tag = wordtagidx
        model = BiLSTM_CRF(len(word2idx), tag2idx, 5, 4,ctx=ctx)
        model.load_params(modelpath,ctx=ctx)
        self.model = model
        self.word2idx = word2idx
        self.idx2tag = idx2tag
        self.ctx = ctx
    def predict(self,string):
        string = list(string)
        precheck_sent = prepare_sequence(string, self.word2idx)
        prd = self.model(precheck_sent.as_in_context(self.ctx))
        n = -1
        word = []
        words = []
        targs = []
        targidx = 0
        for i,zzz in enumerate(prd[1]):
            if zzz <= len(self.idx2tag)-1:
                if n == -1:
                    word.append(string[i])
                    n = i
                    targidx = zzz
                else:
                    if n+1 == i and targidx == zzz:
                        word.append(string[i])
                        n = i
                    else:

                        words.append(''.join(word))
                        targs.append(self.idx2tag[targidx])
                        targidx = zzz
                        word=[]
                        word.append(string[i])
                        n = i
        words.append(''.join(word))
        targs.append(self.idx2tag[targidx])
        return words,targs

import os
ner = NER("model.params",wordtagidx,ctx=mx.cpu())
f = open(sys.argv[1],encoding="utf-8")

string = f.read()
string = string.replace("\n", "")
out = ner.predict(string)
for i,word in enumerate(out[0]):
    print(word + '  '+ out[1][i])
