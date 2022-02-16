import argparse
import warnings
import pandas as pd
from os.path import join as pjoin
from preprocess import processing, split_dataset
from reaction_allocator import allocate_class

warnings.filterwarnings(action='ignore')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build Reaction Dataset')
    parser.add_argument('--split',
                        action='store_true',
                        default=False)

    parser.add_argument('--preprocessing',
                        action='store_true',
                        default=False)

    parser.add_argument('--allocate_reaction',
                        action='store_true',
                        default=False)

    parser.add_argument('--use_textdistance',
                        action='store_true',
                        default=False)

    parser.add_argument('--data_dir',
                        type=str,
                        default='data')

    parser.add_argument('--result_dir',
                        type=str,
                        default='result')

    args = parser.parse_args()
    
    data = pd.read_csv(pjoin(args.data_dir, 'data.csv')).dropna(axis=0)
    if args.preprocessing:
        data = processing(args, data)
    if args.split:
        split_dataset(args, data)

    if args.allocate_reaction:
        allocate_class(args)

    
    