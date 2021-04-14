import torch
from torch.utils.data import Dataset,DataLoader
import json
import numpy as np


class ChatDataset(Dataset):
    def __init__(self,corpus):
        self.corpus = corpus

    def __getitem__(self, i):
        question = torch.LongTensor(self.corpus[i][0])
        reply = torch.LongTensor(self.corpus[i][1])
        return (question, reply)

    def __len__(self):
        return len(self.corpus)


def create_masks(question, reply_input, reply_target,device):
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)

    question_mask = (question != 0).to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

    reply_input_mask = reply_input != 0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data)
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target != 0  # (batch_size, max_words)

    return question_mask, reply_input_mask, reply_target_mask



class TrainValLoader:
    def __init__(self,vars):
        self.train_size =  vars['train_size']
        self.num_workers = vars['num_workers']
        self.batch_size = vars['batch_size']
        self.corpus = self.load_corpus(vars['corpus_encoded_path'])

    @staticmethod
    def load_corpus(json_path):
        with open(json_path, 'r', encoding='utf8') as corpus_file:
            corpus = json.load(corpus_file)
        return corpus

    def get_train_val_loader(self):
        np.random.shuffle(np.array(self.corpus))
        train_len = int(self.train_size * len(self.corpus))
        train_corpus = self.corpus[:train_len]
        val_corpus = self.corpus[train_len:]
        train_dataset = ChatDataset(train_corpus)
        val_dataset = ChatDataset(val_corpus)
        train_loader = DataLoader(ChatDataset(train_corpus),batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)
        val_loader = DataLoader(ChatDataset(val_corpus),batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)
        return train_loader,val_loader


