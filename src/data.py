import io
from collections import Counter
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from .utils import word2subword

def build_vocab(filepath):
    tokenizer = get_tokenizer('spacy', language='en')
    subword_counter, fullword_counter = Counter(), Counter()
    list_of_sentences, all_text = [], []
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            # e.g. ['A', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.', '\n']
            list_of_word_tokens = tokenizer(string_)
            list_of_sentences.append(list_of_word_tokens)
            
            # full text and full word token for skipgram sampling later
            all_text.extend(list_of_word_tokens)
            fullword_counter.update(list_of_word_tokens)
            
            # update subword token
            for word_token in list_of_word_tokens:
                subtokens = word2subword(word_token)
                subword_counter.update(subtokens)
                
    subword_vocab = Vocab(subword_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    fullword_vocab = Vocab(fullword_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    return subword_vocab, fullword_vocab, len(fullword_vocab.stoi), list_of_sentences, all_text

class neg_sampling_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, couples, labels, fullword_vocab):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.couples = couples
        self.labels = labels
        self.fullword_vocab = fullword_vocab
        assert len(couples) == len(labels), f"number of word pairs and labels do not match {len(couples)} vs {len(labels)}"

    def __len__(self):
        return len(self.labels)
    
    def _idx2word(self, idx):
        if type(idx) == int:
            return self.fullword_vocab.itos[idx]
        elif type(idx) == str:
            return idx
        else:
            raise TypeError(f"unrecognized type {type(idx)}")
    
    def __getitem__(self, idx):
        couple = self.couples[idx]
        couple = [self._idx2word(word) for word in couple]
        label = self.labels[idx]
        return couple, label