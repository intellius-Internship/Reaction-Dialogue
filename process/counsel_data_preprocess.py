import re
import pandas as pd
import numpy as np

from functools import reduce

from os.path import join as pjoin

ROOT_DIR = '..'
DATA_DIR = pjoin(ROOT_DIR, 'data/counsel')

def del_newline(text : str):
    return re.sub('[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!~@0-9a-zA-Z\s]+', '', text)

def del_nickname(text : str):
    return re.sub('@[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]*', '', text)

def preprocess(text : str):
    proc_txt = del_newline(text)
    proc_txt = del_special_char(proc_txt)
    proc_txt = del_nickname(proc_txt)

    return proc_txt.strip()

def is_valid(proc_text : str, threshold=2) -> bool:
    return len(proc_text) > threshold

if __name__=="__main__":
    train = pd.read_csv(pjoin(DATA_DIR, 'train.csv')).dropna(axis=0)
    test = pd.read_csv(pjoin(DATA_DIR, 'test.csv')).dropna(axis=0)
    valid = pd.read_csv(pjoin(DATA_DIR, 'valid.csv')).dropna(axis=0)


    data = pd.concat([train, test, valid], ignore_index=True)
    data = data.sample(frac=1)
    
    prev_data_len = len(data)
    data['proc_reply'] = list(map(preprocess, data['reply']))
    data['proc_query'] = list(map(preprocess, data['query']))

    labels = data.intent.unique().tolist()
    
    data['label'] = list(map(lambda x: labels.index(x), data['intent']))
    data.drop_duplicates(['query'], inplace=True, keep='first', ignore_index=True)

    data.to_csv(pjoin(DATA_DIR, 'data.csv'), index=False)
    
    valid_ratio = 0.15
    num_drop = 0

    valid = pd.DataFrame()
    test = pd.DataFrame()
    train = pd.DataFrame()

    for idx in range(len(labels)):
        sub_data = data[data.label==idx]

        num_valid = int(len(sub_data) * valid_ratio)
        if num_valid == 0:
            num_drop += 1
            continue 

        valid = pd.concat([valid, sub_data.iloc[:num_valid]], ignore_index=True)
        test = pd.concat([test, sub_data.iloc[num_valid:2*num_valid]], ignore_index=True)
        train = pd.concat([train, sub_data.iloc[2*num_valid:]], ignore_index=True)
    
    proc_labels = valid.intent.unique().tolist()
    print(proc_labels)
    
    valid['label'] = list(map(lambda x: proc_labels.index(x), valid['intent']))
    test['label'] = list(map(lambda x: proc_labels.index(x), test['intent']))
    train['label'] = list(map(lambda x: proc_labels.index(x), train['intent']))

    valid = valid.sample(frac=1)
    test = test.sample(frac=1)
    train = train.sample(frac=1)

    valid.to_csv(pjoin(DATA_DIR, 'valid.csv'), index=False)
    test.to_csv(pjoin(DATA_DIR, 'test.csv'), index=False)
    train.to_csv(pjoin(DATA_DIR, 'train.csv'), index=False)

    print(f"Drop num: {num_drop} \nLabel num: {len(valid.label.unique())}")

    print(f"Total Number of Data : {prev_data_len} -> {len(valid) + len(test) + len(train)}")
