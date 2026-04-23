from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
nltk.download('punkt')
nltk.download('punkt_tab')


class NewsTokenizer:
    def __init__(self, max_title_len=30, min_word_freq=2):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}


    def build_vocab(self, titles):
        word_counts = Counter()
        for title in titles:
            tokens = nltk.word_tokenize(title.lower())
            word_counts.update(tokens)
        for word, count in word_counts.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f'Vocabulary size: {len(self.word2idx)}')


    def encode_title(self, title):
        tokens = nltk.word_tokenize(title.lower())
        indices = [self.word2idx.get(t, 1) for t in tokens]
        # Pad or truncate
        if len(indices) < self.max_title_len:
            indices += [0] * (self.max_title_len - len(indices))
        else:
            indices = indices[:self.max_title_len]
        return indices


def load_glove(glove_path, word2idx, embed_dim=300):
    print('Loading Glove')
    embedding_matrix = np.random.normal(
        size=(len(word2idx), embed_dim)).astype('float32') * 0.1
    embedding_matrix[0] = 0  # PAD vector
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                embedding_matrix[word2idx[word]] = np.array(parts[1:], dtype='float32')
                found += 1
    print(f'Found {found}/{len(word2idx)} words in GloVe')
    return embedding_matrix


def parse_behaviors(behaviors_df, news_encoded, neg_k=4):
    samples = []
    for _, row in behaviors_df.iterrows():
        history = row['history'].split() if pd.notna(row['history']) else []
        history_encoded = [news_encoded[nid]
                           for nid in history if nid in news_encoded]
        impressions = row['impressions'].split()
        pos = [imp.split('-')[0] for imp in impressions
               if imp.endswith('-1')]
        neg = [imp.split('-')[0] for imp in impressions
               if imp.endswith('-0')]
        for p in pos:
            sampled_neg = np.random.choice(
                neg, size=min(neg_k, len(neg)), replace=False)
            candidates = [p] + list(sampled_neg)
            labels = [1] + [0] * len(sampled_neg)
            samples.append({
                'history': history_encoded[-50:],  # last 50
                'candidates': [news_encoded[c]
                               for c in candidates
                               if c in news_encoded],
                'labels': labels
            })
    return samples

def preprocess():
    news_cols = ['news_id', 'category', 'subcategory', 'title',
                'abstract', 'url', 'title_entities', 'abstract_entities']
    news_df = pd.read_csv('data/MINDsmall_train/news.tsv',
                        sep='\t', names=news_cols)

    # Load behaviors data
    beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv',
                                sep='\t', names=beh_cols)
    
    titles = news_df['title']

    tokenizer = NewsTokenizer()
    tokenizer.build_vocab(titles.values)

    glove_embed = load_glove(r'data/glove/glove.6B.300d.txt', tokenizer.word2idx)

    encoded_titles = dict()

    for _, row in news_df.iterrows():
        news_id = row['news_id']
        title = row['title']
        encoded_titles[news_id] = tokenizer.encode_title(title)

    training_behaviors = parse_behaviors(behaviors_df, encoded_titles)

    news_val_df = pd.read_csv('data/MINDsmall_dev/news.tsv',
                        sep='\t', names=news_cols)

    behaviors_val_df = pd.read_csv('data/MINDsmall_dev/behaviors.tsv',
                                sep='\t', names=beh_cols)

    titles_val = news_val_df['title']

    tokenizer_val = NewsTokenizer()
    tokenizer_val.build_vocab(titles_val.values)

    encoded_titles_val = dict()

    for _, row in news_val_df.iterrows():
        news_id_val = row['news_id']
        title_val = row['title']
        encoded_titles_val[news_id_val] = tokenizer_val.encode_title(title_val)

    val_behaviors = parse_behaviors(behaviors_val_df, encoded_titles_val)

    return training_behaviors, val_behaviors, glove_embed

def collate_fn(max_history, max_title_len, neg_k):
    PAD_TITLE = [0] * max_title_len

    def collate(batch):
        histories, candidates, labels, hist_masks = [], [], [], []
        for sample in batch:
            hist = sample['history'][-max_history:]
            hist_mask = [1] * len(hist) + [0] * (max_history - len(hist))
            hist += [PAD_TITLE] * (max_history - len(hist))

            cands = sample['candidates']
            cands += [PAD_TITLE] * (neg_k + 1 - len(cands))

            lbls = sample['labels'] + [0] * (neg_k + 1 - len(sample['labels']))

            histories.append(torch.tensor(hist))
            candidates.append(torch.tensor(cands[:neg_k+1]))
            labels.append(torch.tensor(lbls[:neg_k+1]))
            hist_masks.append(torch.tensor(hist_mask))

        return {
            'history':    torch.stack(histories),
            'candidates': torch.stack(candidates),
            'labels':     torch.stack(labels),
            'hist_mask':  torch.stack(hist_masks)
        }
    
    return collate

def data_load(batch_size, neg_k, max_history, max_title_len):
    print('Calling Preprocess')
    training_data, val_data, glove_embed = preprocess()

    train_loader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn(max_history, max_title_len, neg_k))

    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn(max_history, max_title_len, neg_k))
    
    return train_loader, val_loader, glove_embed