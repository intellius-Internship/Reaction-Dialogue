import re
import argparse
import random
import pandas as pd
import numpy as np
import warnings
from functools import reduce
from os.path import join as pjoin
warnings.filterwarnings(action='ignore')

repeatchars_pattern = re.compile('(\D)\\1{2,}')
doublespace_pattern = re.compile('\s+')

def repeat_normalize(sent, num_repeats=3):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def del_newline(text : str):
    return re.sub('[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!~0-9a-zA-Z\s]+', '', text)

def del_nickname(text : str):
    return re.sub('@[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]*', '', text)

def preprocess(text : str):
    proc_txt = del_newline(text)
    proc_txt = del_special_char(proc_txt)
    proc_txt = repeat_normalize(proc_txt, num_repeats=3)

    return proc_txt.strip()

def is_valid(proc_text : str, threshold=2) -> bool:
    return len(re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ]', '', proc_text)) > threshold

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/data.csv')

    parser.add_argument('--save_dir',
                        type=str,
                        default='../data')

    parser.add_argument('--seed',
                        type=int,
                        default=19)

    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.2)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = pd.read_csv(args.data_path).dropna(axis=0)
    print(f'Original Length of Data : {len(data)}')
    
    data['proc_query'] = list(map(preprocess, data['query']))
    data['proc_reply'] = list(map(preprocess, data['reply']))
    data.to_csv(pjoin(args.save_dir, 'data.csv'), index=False)

    proc_data = data #[data['is_valid']]

    test = pd.DataFrame()
    train = pd.DataFrame()

    for cls in data.reaction_cls.unique().tolist():
        sub_data = proc_data[proc_data.reaction_cls==cls]
        num_test = int(len(sub_data) * args.test_ratio)

        if num_test == 0:
            if len(sub_data) < 2:
                train = pd.concat([train, sub_data], ignore_index=True)
            else:
                num_test = int(len(sub_data)/2)
                test = pd.concat([test, sub_data.iloc[:num_test]], ignore_index=True)
                train = pd.concat([train, sub_data.iloc[num_test:]], ignore_index=True)
        else:
            test = pd.concat([test, sub_data.iloc[:num_test]], ignore_index=True)
            train = pd.concat([train, sub_data.iloc[num_test:]], ignore_index=True)

        del sub_data

    test = test.sample(frac=1, random_state=args.seed)
    train = train.sample(frac=1, random_state=args.seed)

    test.to_csv(pjoin(args.save_dir, 'test.csv'), index=False)
    train.to_csv(pjoin(args.save_dir, 'train.csv'), index=False)

    print(f"Total Number of Data : {len(data)} -> {len(test) + len(train)}")
