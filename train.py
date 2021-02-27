from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchtext.utils import download_from_url, extract_archive
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from src import build_vocab, subword_tokenizer, neg_sampling_dataset

WINDOW_SIZE = 4
N_EPOCHS = 5
BATCH_SIZE = 50
LEARNING_RATE = 0.005

# 1. Download data
print("1. downloading data")
url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ['train.en.gz']
val_urls = ['val.en.gz']
test_urls = ['test_2016_flickr.en.gz']

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]


# 2. preprocess data and get negative sampling pairs
print("2. preprocessing data")
en_vocab, fullword_vocab, vocab_size, list_of_sentence_tokens, all_text = build_vocab(train_filepaths[0])
print(f"full word vocab size: {vocab_size}")

sampling_table = make_sampling_table(vocab_size)
couples, labels = skipgrams(
      all_text, 
      vocabulary_size=vocab_size,
      window_size=WINDOW_SIZE,
    )

dataset = neg_sampling_dataset(couples, labels, fullword_vocab)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


# 3. init model and train
print("3. training model")
model = subword_tokenizer(en_vocab, 20)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

for epoch in range(N_EPOCHS):
    train_loss, train_acc = 0, 0
    print(f"epoch: {epoch}")
    
    for idx, (couples_batch, labels_batch) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        loss, logits = model.batch_forward(couples_batch, labels_batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (logits.argmax(1) == labels_batch).sum().item()    
    scheduler.step()
    print(f" {train_loss / len(dataset)}, {train_acc / len(dataset)}")

model.save('savedmodel')