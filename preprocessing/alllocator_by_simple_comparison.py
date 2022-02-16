import re
import multiprocessing
import pandas as pd
import numpy as np

from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from functools import partial
from os.path import join as pjoin

def to_list(inputs):
    if isinstance(inputs, str):
        return eval(inputs)
    return inputs

def get_valid_list(x):
    try:
        if len(x) == 1 and len(x[0]) > 0:
            return x[0]
    except TypeError:
        return x
    return x   

def parallelize_dataframe(data, func, num_cores, args):
    sub_data = np.array_split(data, num_cores)
    pool = Pool(num_cores)

    data = pd.concat(pool.map(partial(func, args), sub_data))
    pool.close()
    pool.join()
    return data

def get_core_speech(text):
    return re.sub('[^가-힣ㄱ-ㅎ?]', '', text)

def get_similar_speech(reaction_data : pd.DataFrame, query_reaction : str):
    reaction = reaction_data[reaction_data['reaction_cls']==query_reaction]
    assert len(reaction) == 1
    return reaction['response-list'].tolist()[0]

def get_regexp_list(reaction_data, text):
    return reaction_data[reaction_data.reaction_cls==text]['regexp-list'].tolist()[0]

def get_regexp(reaction_data, text):
    regexp_list = get_regexp_list(reaction_data, text)
    return '|'.join(regexp_list)

def get_candidates_by_regexp(data : pd.DataFrame, reaction_data : pd.DataFrame, react : str) -> list:
    regexp = get_regexp(reaction_data=reaction_data, text=react)
    try:
        candidate_turns = list(filter(lambda x: len(re.findall(regexp, x[-1])) > 0, zip(data['query'], data['reply'])))
    except Exception:
        print(f"Error on react: {react}")

    return candidate_turns

def get_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame, react : str) -> list:
    replies = get_similar_speech(reaction_data=reaction_data, text=react)
    core_replies = list(map(get_core_speech, replies))
    
    candidate_turns = list(filter(lambda text: reduce(lambda x, y: x|y, list(map(lambda x: x in text[0], core_replies))), \
        zip(data.core_reply, data['query'], data['reply'])))
    candidate_turns = list(map(lambda x: (x[1], x[-1]), candidate_turns))
    return candidate_turns

def allocate_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame) -> pd.DataFrame:
    replies_by_react = []
    for react in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique())):
        candidate_turns = get_candidates_by_regexp(data=data, reaction_data=reaction_data, react=react)
        replies_by_react.append(candidate_turns)

    reaction_data['candidate_turns'] = replies_by_react
    reaction_data['num_candidate'] = list(map(lambda x: len(x), reaction_data['candidate_turns']))
    return reaction_data

def postprocess_candidates(reaction_data):
    entire_data = pd.DataFrame()
    for react in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique())):
        sub_reaction_data = reaction_data[reaction_data.reaction_cls==react]
        candidate_turns = sub_reaction_data['candidate_turns'].tolist()[0]

        sub_data = pd.DataFrame()
        sub_data['query'] = list(map(lambda x: x[0], candidate_turns))
        sub_data['reply'] = list(map(lambda x: x[-1], candidate_turns))
        sub_data['reaction_cls'] = react
        entire_data = pd.concat([entire_data, sub_data], ignore_index=True)
        del sub_data, candidate_turns

    return entire_data

def allocate_class_by_simple_comparison(args, reaction_data, data):
    # data['core_reply'] = list(map(get_core_speech, data['reply']))

    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")

    reaction_data = parallelize_dataframe(reaction_data, allocate_candidates, \
        num_cores=num_cores, args=data)
    reaction_data.to_csv(pjoin(args.result_dir, 'reaction_with_candidates.csv'), index=False)

    entire_data = postprocess_candidates(reaction_data)
    entire_data.to_csv(pjoin(args.result_dir, 'entire_data.csv'), index=False)
    return entire_data