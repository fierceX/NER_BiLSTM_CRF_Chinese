import mxnet as mx
from mxnet import autograd as ag, ndarray as nd, gluon
from mxnet.gluon import Block, nn, rnn
import mxnet.optimizer as optim
import sys


def to_scalar(x):
    return int(x.asscalar())

def argmax(vec):
    idx = nd.argmax(vec, axis=1)
    return to_scalar(idx)

def prepare_sequence(seq, word2idx):
    return nd.array([word2idx[w] for w in seq])


def log_sum_exp(vec):
    max_score = nd.max(vec).asscalar()
    return nd.log(nd.sum(nd.exp(vec - max_score))) + max_score

class BiLSTM_CRF(Block):
    def __init__(self, vocab_size, tag2idx, embedding_dim, hidden_dim,START_TAG = "<START>",STOP_TAG = "<STOP>",ctx=mx.cpu()):
        super(BiLSTM_CRF, self).__init__()
        with self.name_scope():
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.vocab_size = vocab_size
            self.tag2idx = tag2idx
            self.START_TAG = START_TAG
            self.STOP_TAG = STOP_TAG
            self.tagset_size = len(tag2idx)
            
            self.ctx = ctx

            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, num_layers=1, bidirectional=True)

            self.hidden2tag = nn.Dense(self.tagset_size)

            self.transitions = nd.random.normal(shape=(self.tagset_size, self.tagset_size),ctx=ctx)

            self.hidden = self.init_hidden()
            

    def init_hidden(self):
        return [nd.random.normal(shape=(2, 1, self.hidden_dim // 2),ctx=self.ctx),
                nd.random.normal(shape=(2, 1, self.hidden_dim // 2),ctx=self.ctx)]

    def _forward_alg(self, feats):
        alphas = [[-10000.] * self.tagset_size]
        alphas[0][self.tag2idx[self.START_TAG]] = 0.
        alphas = nd.array(alphas,ctx=self.ctx)

        for feat in feats:
            alphas_t = [] 
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].reshape((1, -1))
                trans_score = self.transitions[next_tag].reshape((1, -1))
                next_tag_var = alphas + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var))
            alphas = nd.concat(*alphas_t, dim=0).reshape((1, -1))
        terminal_var = alphas + self.transitions[self.tag2idx[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[0]
        embeds = self.word_embeds(sentence).reshape((length, 1, -1))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.reshape((length, self.hidden_dim))
        lstm_feats = self.hidden2tag(lstm_out)
        return nd.split(lstm_feats, num_outputs=length, axis=0, squeeze_axis=True)

    def _score_sentence(self, feats, tags):
        score = nd.array([0],ctx=self.ctx)
        tags = nd.concat(nd.array([self.tag2idx[self.START_TAG]],ctx=self.ctx), *tags, dim=0)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[to_scalar(tags[i+1]), to_scalar(tags[i])] + feat[to_scalar(tags[i+1])]
        score = score + self.transitions[self.tag2idx[self.STOP_TAG],
                                         to_scalar(tags[int(tags.shape[0]-1)])]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        vvars = nd.full((1, self.tagset_size), -10000.,ctx=self.ctx)
        vvars[0, self.tag2idx[self.START_TAG]] = 0

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):

                next_tag_var = vvars + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0, best_tag_id])

            vvars = (nd.concat(*viterbivars_t, dim=0) + feat).reshape((1, -1))
            backpointers.append(bptrs_t)

        terminal_var = vvars + self.transitions[self.tag2idx[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0, best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2idx[self.START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq