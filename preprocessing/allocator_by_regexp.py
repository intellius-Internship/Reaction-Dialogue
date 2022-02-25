import os
import re
import multiprocessing
import pandas as pd
import numpy as np

from tqdm import tqdm
from os.path import join as pjoin
from util import parallelize_dataframe

'''
Description
-----------
주어진 리액션 클래스의 정규식 반환
'''
def get_regexp_list(reaction_data, reaction_cls : str):
    return reaction_data[reaction_data.reaction_cls==reaction_cls]['regexp-list'].tolist()[0]

def get_regexp(reaction_data, reaction_cls : str):
    regexp_list = get_regexp_list(reaction_data=reaction_data, reaction_cls=reaction_cls)
    return '|'.join(regexp_list)

'''
Description
-----------
주어진 리액션 클래스의 정규식에 포함되는 응답을 가진 대화 턴을 후보 턴으로 저장 
'''
def get_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame, reaction_cls : str) -> list:
    regexp = get_regexp(reaction_data=reaction_data, reaction_cls=reaction_cls)
    try:
        candidate_turns = list(filter(lambda x: len(re.findall(regexp, x[-1])) > 0, zip(data['query'], data['reply'])))
    except Exception:
        print(f"Error on react: {reaction_cls}")

    return candidate_turns

def allocate_candidates(data : pd.DataFrame, reaction_data : pd.DataFrame) -> pd.DataFrame:
    replies_by_react = []

    # integrate all candidate dataset
    for reaction_cls in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique()),
        desc=f"PID: {os.getpid()}", mininterval=0.01):
        candidate_turns = get_candidates(data=data, reaction_data=reaction_data, reaction_cls=reaction_cls)
        # '아니' 클래스의 경우, query가 의문문인지 확인
        if reaction_cls == '아니':
            temp_candidates = list(filter(lambda text: len(re.findall('^[아-잏하-힣]{0,2}[,.~?!ㄱ-ㅎㅏ-ㅣ\s]*아[니-닣]', text[-1])) > 0, 
                                        zip(data['query'], data['reply'])))
            temp_candidates = list(filter(lambda text: len(re.findall('(\?|[냐-냫노-놓누-눟녀-녛니-닣])$', text[0])) > 0, \
                                                        temp_candidates))
            candidate_turns += temp_candidates
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
    for react in tqdm(reaction_data.reaction_cls.unique(), total=len(reaction_data.reaction_cls.unique()), \
        desc=f"postprocess", mininterval=0.01):

        sub_reaction_data = reaction_data[reaction_data.reaction_cls==react]
        candidate_turns = sub_reaction_data['candidate_turns'].tolist()[0]

        sub_data = pd.DataFrame()
        sub_data['query'] = list(map(lambda x: x[0], candidate_turns))
        sub_data['reply'] = list(map(lambda x: x[-1], candidate_turns))
        sub_data['reaction_cls'] = react
        entire_data = pd.concat([entire_data, sub_data], ignore_index=True)
        del sub_data, candidate_turns

    return entire_data

def allocate_class_by_regexp(args, reaction_data, data):
    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")

    # multiprocessing
    reaction_data = parallelize_dataframe(reaction_data, allocate_candidates, \
        num_cores=num_cores, args=data)

    entire_data = postprocess_candidates(reaction_data)
    entire_data.to_csv(pjoin(args.result_dir, 'labeled_data_by_regexp.csv'), index=False)
    return 