import textdistance
import pandas as pd

from tqdm import tqdm
from os.path import join as pjoin
from typing import List

'''
Description
-----------
텍스트 유사도 기반 라벨링에 필요한 파라미터 지정

    args.threshold: 텍스트 유사도 기반 유사도 점수 임계값 
    args.algorithm: 텍스트 유사도 알고리즘
'''
def base_setting(args):
    args.threshold = getattr(args, 'threshold', 0.8)
    args.algorithm = getattr(args, 'algorithm', 'jaro_winkler')


'''
Description
-----------
텍스트 유사도 기반 유사도 점수 계산 (0 to 1)
'''
def get_similar_score(hypo, ref, algorithm):
    if algorithm == 'jaro_winkler':
        return 1-textdistance.jaro_winkler.normalized_distance(hypo, ref)
    if algorithm == 'levenshtein':
        return 1-textdistance.levenshtein.normalized_distance(hypo, ref)
    if algorithm == 'hamming':
        return 1-textdistance.hamming.normalized_distance(hypo, ref)
    
    raise NotImplementedError('Not Implemented')

'''
Description
-----------
리액션 응답인 reply와 유사한 응답을 가지는 대화 턴 후보 추출
'''
def get_candidate_turns(data, reply, threshold : float, algorithm : str):
    cand_turn = list(map(lambda x: (get_similar_score(x[0], reply, algorithm), x[0], x[1]), zip(data['query'], data['reply'])))
    cand_turn = sorted(cand_turn, key=lambda x: x[0], reverse=True)
    cand_turn = list(filter(lambda x: x[0] > threshold, cand_turn))
    return cand_turn

'''
Description
-----------
reaction 클래스의 유사 응답인 replies와 \
    유사한 응답을 가지는 대화 턴 후보 추출 및 dataframe 생성
'''
def get_candidates(args, data, reaction, replies : List[str]):
    entire_candidates = pd.DataFrame()
    for reply in tqdm(replies, total = len(replies), desc=f'process: {reaction}'):
        turns = get_candidate_turns(data=data, reply=reply, threshold=args.threshold, algorithm=args.algorithm)
            
        sub_candidates = pd.DataFrame()
        sub_candidates['score'] = list(map(lambda x: x[0], turns))
        sub_candidates['query'] = list(map(lambda x: x[1], turns))
        sub_candidates['reply'] = list(map(lambda x: x[-1], turns))

        entire_candidates = pd.concat([entire_candidates, sub_candidates], ignore_index=True)
        del sub_candidates, turns
        
    entire_candidates['reaction'] = reaction
    return entire_candidates

def allocate_candidates(args, data, reaction_data):
    entire_data = pd.DataFrame()
    for d in tqdm(reaction_data.iterrows(), total = len(reaction_data), desc='extract candidates'):
        row = d[1]
        replies = list(set(row['reply'] + [row['reaction']]))
        candidates = get_candidates(args=args, data=data, reaction=row['reaction'], replies=replies)

        entire_data = pd.concat([entire_data, candidates], ignore_index=True)
        del replies, candidates
    return entire_data

'''
Description
-----------
각 대화 턴의 후보 클래스 중 가장 높은 스코어를 가지는 클래스로 라벨링
'''
def allocate_reaction(data : pd.DataFrame):
    result = pd.DataFrame()
    data_wo_duplicates = data.drop_duplicates(['query', 'reply'])

    for d in tqdm(data_wo_duplicates.iterrows(), total=len(data_wo_duplicates), desc='labeling reaction'):
        row = d[1]
        sub_data = data[(data['query'] == row['query']) & (data['reply'] == row['reply'])]
        sub_data.sort_values(by=['score'], axis=0, inplace=True, ascending=False)
        result = pd.concat([result, sub_data.iloc[:1]], ignore_index=True)
        
    return result

def allocate_class_by_textdistance(args, data, reaction_data):
    base_setting(args)

    data = allocate_candidates(args, data, reaction_data)
    data = allocate_reaction(data)
    data.to_csv(pjoin(args.result_dir, 'labeled_data_by_textdist.csv'))

    return