
# -*- coding: utf-8 -*-
import re
import torch
import pandas as pd

from torch import nn
from os.path import join as pjoin
from plm import LightningPLM
from utils.model_util import load_model

def base_setting(args):
    args.max_len = getattr(args, 'max_len', 512)
    args.batch_size = getattr(args, 'batch_size', 4)
    args.log = getattr(args, 'log', True)

def tokenize(tokenizer, text, max_len):
    q_toked = tokenizer.tokenize(tokenizer.cls_token + text + tokenizer.sep_token)
    if len(q_toked) > max_len:
        q_toked = q_toked[:max_len-1] + q_toked[-1]

    token_ids = tokenizer.convert_tokens_to_ids(q_toked)
    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]
        attention_mask += [0]

    return token_ids, attention_mask


def evaluation(args, **kwargs):
    # load params
    base_setting(args)
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    print(args.model_pt)

    model, tokenizer = load_model(args.model_type, args.num_labels)
    model = model.cuda()

    if args.model_pt is not None:
        if args.model_pt.endswith('ckpt'):
            model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
        else:
            raise TypeError('Unknown file extension')

    model = model.cuda()     
    model.eval()

    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    test_data = test_data.dropna(axis=0)

    pred_list = []

    count = 0
    with torch.no_grad():
        for row in test_data.iterrows():

            utterance = row[1]['proc_query']
            label = int(row[1]['label'])
            
            input_ids, attention_mask = tokenize(tokenizer, text=utterance, max_len=args.max_len)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            print(predictions)
            pred_list.append(predictions[0]) 

            if predictions[0] == label:
                count += 1

        test_data['pred'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'{args.model_name}.csv'), sep='\t', index=False)
        print(f"Accuracy: {count/len(test_data)}")
            

