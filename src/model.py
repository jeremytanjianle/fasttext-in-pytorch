import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn 
from .utils import word2subword

class subword_tokenizer(nn.Module):
    """
    USAGE:
    ------
    # Imeplemented model gives the same emebddings
    from src.model import Subword_Embedding
    subword = Subword_Embedding()
    subword.from_pretrained( pretraining_folder = 'resources/cc.en.300.bin' )

    # see something vec
    something_vec = subword(['something', 'something is right'])
    something_vec[:,:5]
    tensor([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
            [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]])

    # saving and restoring function works
    subword.save('test')
    subword2 = Subword_Embedding()
    subword2.restore('test')

    # see something vec
    something_vec = subword2(['something', 'something is right'])
    something_vec[:,:5]
    tensor([[-0.0045,  0.0097,  0.0500,  0.0337, -0.0330],
            [ 0.0011,  0.0044,  0.0108,  0.0488, -0.0035]],
        grad_fn=<SliceBackward>)
    """
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
            # get token ids
            token_id = self.str2id(word)
            token_ids.extend(token_id)
            # store offsets for embedding 
            offsets.append(offset)
            offset += len(token_id)
        token_ids = torch.LongTensor(token_ids)
        
        # feed the ids into embedding
        encoding = self.embedding(token_ids, offsets=torch.tensor(offsets))
        return encoding
        
    def train_batch(self, batch_of_couples, labels):
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
        