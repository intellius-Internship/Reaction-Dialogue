import pandas as pd

from ast import literal_eval
from os.path import join as pjoin

from alllocator_by_simple_comparison import allocate_class_by_simple_comparison
from allocator_by_textdistance import allocate_class_by_textdistance


def allocate_class(args):
    data = pd.read_csv(pjoin(args.data_dir, 'data.csv'))
    reaction_data = pd.read_csv(pjoin(args.data_dir, 'reaction.csv'), converters={
        "response-list": literal_eval,
        "regexp-list" : literal_eval
    })

    if args.use_textdistance:
        allocate_class_by_textdistance(args=args, data=data, reaction_data=reaction_data)
    else:
        allocate_class_by_simple_comparison(args=args, data=data, reaction_data=reaction_data)

    return