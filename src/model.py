import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn 
from .utils import word2subword

class subword_tokenizer(nn.Module):
    def __init__(self, stoi={'<unk>':0}, embed_dim=200):
        super().__init__()
        self.stoi = stoi
        self.embed_dim = embed_dim
        self.vocab_size = len(self.stoi)
        # self.max_len = max_len
        
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim * 2, 2)
        self.init_weights()
        self.criterion = torch.nn.CrossEntropyLoss() #.to(device)
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def str2id(self, text):
        # get token ids
        subword_tokens = word2subword(text)
        subword_tokens_id = [self.stoi[subword_token] for subword_token in subword_tokens]
        return subword_tokens_id
        
    def forward(self, list_of_words):
        if type(list_of_words)==str:
            list_of_words=[list_of_words]
        
        # form a 1d array of the wordpiece ids
        # store a list of offsets to separate the words out
        token_ids, offsets, offset = [], [], 0
        for word in list_of_words:
            offsets.append(offset)
            token_id = self.str2id(word)
            token_ids.extend(token_id)
            offset += len(token_id)
        token_ids = torch.LongTensor(token_ids)
        
        # feed the ids into embedding
        encoding = self.embedding(token_ids, offsets=torch.tensor(offsets))
        return encoding
        
    def batch_forward(self, batch_of_couples, labels):
        # get word pair encodings
        word1s = self(batch_of_couples[0])
        word2s = self(batch_of_couples[1])
        word12_concat = torch.cat((word1s, word2s), axis=1)
        
        # get loss from labels
        labels = torch.tensor(labels)
        logits = self.fc(word12_concat)
        loss = self.criterion(logits, labels)
        
        return loss, logits

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save({'model':self.state_dict(), 
                    'embed_dim':self.embed_dim, 
                    'stoi':self.stoi}, 
                   path+'/state.pt')
        
    def restore(self, path):
        state = torch.load(path+'/state.pt')
        
        # 1. load vocab
        self.stoi = defaultdict(int,state['stoi'])
        self.vocab_size = len(self.stoi)
        
        # 2. load model params
        self.embed_dim = state['embed_dim']
        
        # 3. reinit the layers and load weights
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim, sparse=True)
        self.fc = nn.Linear(self.embed_dim * 2, 2)
        self.load_state_dict(state['model'])
        