import os
import re
import multiprocessing
import pandas as pd

from tqdm.auto import tqdm as a_tqdm
from tqdm import tqdm

from functools import reduce
from util import parallelize_dataframe
from os.path import join as pjoin

'''
Description
-----------
한글과 물음표를 제외한 모든 문제 제거
'''
def get_core_speech(text):
    return re.sub('[^가-힣ㄱ-ㅎ?]', '', text)

def get_keywords(reaction_data : pd.DataFrame, reaction_cls : str):
    reaction = reaction_data[reaction_data['reaction_cls']==reaction_cls]
    assert len(reaction) == 1
    return reaction['response-list'].tolist()[0]

'''
Description
-----------
키워드와 응답의 공백/특수문자/숫자/영문자를 삭제하여 키워드 매칭 여부를 판단하며
응답 내 키워드가 포함된 경우, 해당 대화 턴을 후보 턴으로 저장
'''
def get_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame, reaction_cls : str) -> list:
    keywords = get_keywords(reaction_data=reaction_data, reaction_cls=reaction_cls)
    proc_keywords = list(map(get_core_speech, keywords))
    
    candidate_turns = list(filter(lambda text: reduce(lambda x, y: x|y, list(map(lambda x: x in text[0], proc_keywords))), \
        zip(data['core_reply'], data['query'], data['reply'])))
    candidate_turns = list(map(lambda x: (x[1], x[-1]), candidate_turns))
    return candidate_turns

def allocate_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame) -> pd.DataFrame:
    replies_by_react = []
    for reaction_cls in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique()), \
        desc=f"PID: {os.getpid()}", mininterval=0.01):
        candidate_turns = get_candidates(data=data, reaction_data=reaction_data, reaction_cls=reaction_cls)
        replies_by_react.append(candidate_turns)

    reaction_data['candidate_turns'] = replies_by_react
    reaction_data['num_candidate'] = list(map(lambda x: len(x), reaction_data['candidate_turns']))
    return reaction_data

'''
Description
-----------
후처리 함수로 데이터 구조 변경
'''
def postprocess_candidates(reaction_data):
    entire_data = pd.DataFrame()
    for reaction_cls in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique()), \
            desc=f"postprocess", mininterval=0.01):

        sub_reaction_data = reaction_data[reaction_data.reaction_cls==reaction_cls]
        candidate_turns = sub_reaction_data['candidate_turns'].tolist()[0]

        sub_data = pd.DataFrame()
        sub_data['query'] = list(map(lambda x: x[0], candidate_turns))
        sub_data['reply'] = list(map(lambda x: x[-1], candidate_turns))
        sub_data['reaction_cls'] = reaction_cls
        entire_data = pd.concat([entire_data, sub_data], ignore_index=True)
        del sub_data, candidate_turns

    return entire_data

def allocate_class_by_keyword(args, reaction_data, data):
    # delete spaces, special characters, alphabets and numbers
    a_tqdm.pandas(desc='preprocessing utterance')
    data['core_reply'] = data['reply'].progress_apply(get_core_speech)

    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")

    reaction_data = parallelize_dataframe(reaction_data, allocate_candidates, \
        num_cores=num_cores, args=data)

    entire_data = postprocess_candidates(reaction_data)
    entire_data.to_csv(pjoin(args.result_dir, 'labeled_data_by_kw.csv'), index=False)
    return 