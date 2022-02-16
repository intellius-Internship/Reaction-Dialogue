import numpy as np
import pandas as pd
import warnings

from torch.utils.data import Dataset
from utils.model_utills import U_TKN, S_TKN

warnings.filterwarnings(action='ignore')

DELIMITER = '<unused1>'

class AutoRegressionChatData(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self._data = pd.read_csv(data_path, sep=',')
        self._data = self._data.dropna(axis=0)
        
        self.usr_token = U_TKN
        self.sys_token = S_TKN
        self.delimiter = DELIMITER

        self.first = True
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens, len(tokens)

    def _tokenize_turn(self, query, reply):
        query_toked, query_len = self._tokenize(self.usr_token + str(query))
        reply_toked, reply_len = self._tokenize(self.sys_token + str(reply) + self.tokenizer.eos_token)
        
        if query_len + reply_len > self.max_len:
            remain = self.max_len - query_len
            if remain <= 0:
                # Query가 max_len을 넘어가는 경우, max_len의 반절로 제한
                query_toked = [query_toked[0]] + query_toked[-(int(self.max_len/2))+1:] 
                query_len = len(query_toked)
                remain = self.max_len - query_len
                assert remain > 0

            reply_toked = reply_toked[:remain-1]+[reply_toked[-1]]
            reply_len = len(reply_toked)

        return query_toked, reply_toked, query_len, reply_len
        
    def _padding(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(ids) < self.max_len:
            ids += [self.tokenizer.pad_token_id]
        
        return ids

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        
        query = turn['proc_query']
        reply_cls = turn['reaction_cls'] if 'reaction_cls' in self._data.columns else None
        reply = turn['proc_reply']
        if reply_cls is not None:
            reply = str(reply) + self.delimiter + str(reply_cls)

        query_toked, reply_toked, query_len, reply_len = self._tokenize_turn(query, reply)
        
        labels = [
            self.tokenizer.mask_token,
        ] * query_len + reply_toked[1:]

        labels_ids = self._padding(labels)
        token_ids = self._padding(query_toked + reply_toked)
        mask = [0] * query_len + [1] * reply_len + [0] * (self.max_len - query_len - reply_len)

        return(token_ids, np.array(mask), labels_ids)

class Seq2SeqChatData(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128) -> None:
        self._data = pd.read_csv(data_path, sep=',')
        self._data = self._data.dropna(axis=0)

        self.first = True
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.delimiter = DELIMITER

    def __len__(self):
        return len(self._data)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_len:
            while len(input_id) < self.max_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        turn = self._data.iloc[index]
        
        query = turn['proc_query']
        reply_cls = turn['reaction_cls'] if 'reaction_cls' in self._data.columns else None
        reply = turn['proc_reply']
        if reply_cls is not None:
            reply = reply_cls+self.delimiter+reply
        
        query_toked = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(query) + [self.tokenizer.eos_token]
        reply_toked = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(reply) + [self.tokenizer.eos_token]

        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            query_toked, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            reply_toked, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            reply_toked[1:(self.max_len + 1)])

        if len(labels) < self.max_len:
            while len(labels) < self.max_len:
                labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}



