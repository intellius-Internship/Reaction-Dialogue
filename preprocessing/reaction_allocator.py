import pandas as pd

from ast import literal_eval
from os.path import join as pjoin

from allocator_by_regexp import allocate_class_by_regexp
from allocator_by_textdistance import allocate_class_by_textdistance
from allocator_by_keyword import allocate_class_by_keyword

'''
Description
-----------
리액션 라벨링 함수
    def allocate_class_by_textdistance \
        -> 텍스트 유사도 기반 리액션 라벨링
    def allocate_class_by_regexp \
        -> 정규식 매칭 기반 리액션 라벨링
    def allocate_class_by_keyword \
        -> 키워드 기반 리액션 라벨링
'''
def allocate_class(args):
    # raw dialogue dataset
    data = pd.read_csv(pjoin(args.data_dir, 'data.csv'))

    # reaction-regexp dataset
    reaction_data = pd.read_csv(pjoin(args.data_dir, 'reaction.csv'), converters={
        "response-list": literal_eval,
        "regexp-list" : literal_eval
    })

    if args.labeling == 'textdist':
        # pingpong reaction dataset (only for labeling based on textdistance)
        pingpong_reaction_data = pd.read_csv(pjoin(args.data_dir, 'pingpong_reaction.csv'), converters={
            "reply": literal_eval 
        })
        allocate_class_by_textdistance(args=args, data=data, reaction_data=pingpong_reaction_data)
    elif args.labeling == 'regexp':
        allocate_class_by_regexp(args=args, data=data, reaction_data=reaction_data)
    elif args.labeling == 'keyword':
        allocate_class_by_keyword(args=args, data=data, reaction_data=reaction_data)

    return