import json
import tornado.ioloop
import tornado.web
import sys

import numpy as np
import torch

import math
from torch.utils.data import Dataset
import jieba

from mingpt.utils import top_k_logits
from torch.nn import functional as F


class WordDataset(Dataset):

    def __init__(self, data, block_size, topK=10000):
        words = jieba.lcut(data)
        word_stats = {w: 0 for w in set(words)}
        for w in words:
            word_stats[w] += 1

        # reserve for unknown
        stoi = {'<unk>': 0}
        itos = {0: '<unk>'}

        i = 1
        ditched_set = set()
        for (key, cnt) in sorted(word_stats.items(), key=lambda item: item[1], reverse=True):
            if i > topK:
                ditched_set.add(key)
            else:
                stoi[key] = i
                itos[i] = key
                i += 1

        words = list(map(lambda w: '<unk>' if w in ditched_set else w, words))

        data_size, vocab_size = len(words), i - 1

        print('data has %d words, %d unique words.' % (len(words), vocab_size))

        self.stoi = stoi
        self.itos = itos
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = words

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


@torch.no_grad()
def sample_with_probs(model, x, steps, temperature=1.0, sample=False, top_k=None, first_sentence=False):
    """
    A modified version of sample from mingpt.util that also returns probability of each sentence
    and allow end with first sentence complete
    """
    block_size = model.get_block_size()
    model.eval()
    all_probs = []
    current_prob = 1
    prev_newline = False
    early_terminate = True
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

        # update sentence prob
        curr_newline = train_dataset.itos[ix.item()] == '\n'
        if prev_newline:
            if curr_newline:
                all_probs.append(current_prob)
                if first_sentence:
                    if not early_terminate:
                        early_terminate = True
                    else:
                        break
                current_prob = 1.0
            prev_newline = False
        else:
            if curr_newline:
                prev_newline = True
            current_prob *= probs[0][ix[0][0]].item()

    return x, all_probs


def predict(msgs, cnt=32, input_skip_unk=False, first_sentence=False):
    context = jieba.lcut(msgs)
    x = [0 if s not in train_dataset.stoi else train_dataset.stoi[s] for s in context]
    if input_skip_unk:
        x = list(filter(lambda x: x > 0, x)) # remove <unk>
    x = torch.tensor(x, dtype=torch.long)[None, ...]
    y, all_probs = sample_with_probs(model, x, cnt, temperature=0.9, sample=True, top_k=5,
                                     first_sentence=first_sentence)
    y = y[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion.replace('\n\n', '/'))
    print(all_probs)
    return completion, all_probs


# request handler
class MainHandler(tornado.web.RequestHandler):
    def post(self):
        data_string = self.request.body
        data = json.loads(data_string)
        # use \n\n as sentence separator
        msgs = '\n\n'.join(data['msgs']) + '\n\n'
        resp, all_probs = predict(msgs, cnt=data.get('cnt', 32), input_skip_unk=data.get('input_skip_unk', None),
                                  first_sentence=data.get('first_sentence', False))

        content = json.dumps({'predict': resp, 'probs': all_probs}).encode('utf_8')

        self.set_header("Content-Type", "application/json")
        self.set_header("Content-Length", int(len(content)))
        self.write(content)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('need port number')
        exit(1)
    (model, train_dataset) = torch.load('model.pkl', map_location=torch.device('cpu'))
    app = tornado.web.Application([
        (r"/", MainHandler),
    ])
    app.listen(int(sys.argv[1]))
    tornado.ioloop.IOLoop.current().start()
