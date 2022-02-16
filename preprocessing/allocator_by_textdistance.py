import textdistance
import pandas as pd

from ast import literal_eval
from tqdm import tqdm
from os.path import join as pjoin

def base_setting(args):
    args.threshold = getattr(args, 'threshold', 0.8)
    args.algorithm = getattr(args, 'algorithm', 'jaro_winkler')

def get_similar_score(hypo, ref, algorithm):
    if algorithm == 'jaro_winkler':
        return 1-textdistance.jaro_winkler.normalized_distance(hypo, ref)
    if algorithm == 'levenshtein':
        return 1-textdistance.levenshtein.normalized_distance(hypo, ref)
    if algorithm == 'hamming':
        return 1-textdistance.hamming.normalized_distance(hypo, ref)
    
    raise NotImplementedError('Not Implemented')

def get_similar_turns(data, reply, threshold : float, algorithm : str):
    similar_turn = list(map(lambda x: (get_similar_score(x[0], reply, algorithm), x[0], x[1]), zip(data['query'], data['reply'])))
    similar_turn = sorted(similar_turn, key=lambda x: x[0], reverse=True)
    similar_turn = list(filter(lambda x: x[0] > threshold, similar_turn))
    return similar_turn

def get_candidates(args, data, reaction, replies):
    entire_candidates = pd.DataFrame()
    for reply in tqdm(replies, total = len(replies)):
        turns = get_similar_turns(data=data, reply=reply, threshold=args.threshold, algorithm=args.algorithm)
            
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
    for d in tqdm(reaction_data.iterrows(), total = len(reaction_data)):
        row = d[1]
        replies = row.reply + [row.reaction]
        candidates = get_candidates(args=args, data=data, reaction=row.reaction, replies=replies)

        entire_data = pd.concat([entire_data, candidates], ignore_index=True)
        del replies, candidates
    return entire_data

def allocate_reaction(data : pd.DataFrame):
    result = pd.DataFrame()
    data_wo_duplicates = data.drop_duplicates(['query', 'reply'])

    for d in tqdm(data_wo_duplicates.iterrows(), total=len(data_wo_duplicates)):
        row = d[1]
        sub_data = data[(data['query'] == row.query) & (data.reply == row.reply)]
        sub_data.sort_values(by=['score'], axis=0, inplace=True, ascending=False)
        result = pd.concat([result, sub_data.iloc[:1]], ignore_index=True)
        
    return result

def allocate_class_by_textdistance(args):
    base_setting(args)

    data = pd.read_csv(pjoin(args.data_dir, 'data.csv'))
    reaction_data = pd.read_csv(pjoin(args.data_dir, 'reaction_reply'), converters={
        "reply": literal_eval 
    })

    data = allocate_candidates(args, data, reaction_data)
    data = allocate_reaction(data)
    data.to_csv(pjoin(args.result_dir, 'data.csv'))

    return