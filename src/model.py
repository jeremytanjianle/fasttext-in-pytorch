import os
import numpy as np
import torch
from torch import nn

def get_hash(subword, bucket=2000000, nb_words=2000000):
    h = 2166136261
    for c in subword:
        c = ord(c) % 2**8
        h = (h ^ c) % 2**32
        h = (h * 16777619) % 2**32
    return h % bucket + nb_words

def get_subwords(word, vocabulary, minn=5, maxn=5):
    _word = "<" + word + ">"
    _subwords = []
    _subword_ids = []
    if word in vocabulary:
        _subwords.append(word)
        _subword_ids.append(vocabulary.index(word))
        if word == "</s>":
            return _subwords, np.array(_subword_ids)
    for ngram_start in range(0, len(_word)):
        for ngram_length in range(minn, maxn+1):
            if ngram_start+ngram_length <= len(_word):
                _candidate_subword = _word[ngram_start:ngram_start+ngram_length]
                if _candidate_subword not in _subwords:
                    _subwords.append(_candidate_subword)
                    _subword_ids.append(get_hash(_candidate_subword))
    return _subwords, np.array(_subword_ids)

def load_embeddings(output_dir):
    input_matrix = np.load(os.path.join(output_dir, "embeddings.npy"))
    words = []
    with open(os.path.join(output_dir, "vocabulary.txt"), "r", encoding='utf-8') as f:
        for line in f.readlines():
            words.append(line.rstrip())
    return words, input_matrix

class Subword_Embedding(nn.Module):
    
    def __init__(self, pretrained_source=None):
        super(Subword_Embedding, self).__init__()
        if pretrained_source: self.from_pretrained(pretrained_source)
        
    def from_pretrained(self, pretrained_source):
        self.vocabulary, np_embeddings = load_embeddings(pretrained_source)
        vocab_len, word_dim = np_embeddings.shape
        self.embedding = nn.EmbeddingBag.from_pretrained(torch.FloatTensor(np_embeddings))
        
    def forward(self, words):
        subwords = [get_subwords(word, self.vocabulary) for word in words]
        subword_idx_by_words = [subword[1] for subword in subwords]
        
        subword_idx = self.get_subword_idx(subword_idx_by_words)
        offsets = self.get_offsets(subword_idx_by_words)
        
        embeddings_of_words = self.embedding(subword_idx, offsets)
        
        return embeddings_of_words
        
    def get_subword_idx(self, subword_idx_by_words):
        subword_idx = np.concatenate(subword_idx_by_words)
        subword_idx = torch.tensor(subword_idx, dtype=torch.long)
        return subword_idx
    
    def get_offsets(self, subword_idx_by_words):
        """
        Get offsets of subwords for input into EmbeddingBag
        """
        # collect offsets 
        offsets, running_offset = [0], 0
        for subword_idx_by_word in subword_idx_by_words:
            running_offset += len(subword_idx_by_word)
            offsets.append(running_offset)

        # last offset is not need, as we count them from the front
        offsets = offsets[:-1]
        offsets = torch.tensor(offsets, dtype=torch.long)

        return offsets
