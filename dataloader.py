
import numpy as np
import pandas as pd

from ast import literal_eval
from torch.utils.data import Dataset

class DialogueData(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self._data = pd.read_csv(data_path, sep=',')
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(self.tokenizer.cls_token + str(text) + self.tokenizer.sep_token)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids, len(ids)

    def _padding(self, ids):
        while len(ids) < self.max_len:
            ids += [self.tokenizer.pad_token_id]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len-1] + ids[-1]
        return ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        
        utter = turn['proc_query']
        label = int(turn['label'])

        token_ids, ids_len = self._tokenize(utter)
        token_ids = self._padding(token_ids)

        attention_masks = [float(id>0) for id in token_ids]
        return(token_ids, np.array(attention_masks), label)
