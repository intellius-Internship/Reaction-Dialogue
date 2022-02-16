
# -*- coding: utf-8 -*-
import torch
import re
import pandas as pd
from os.path import join as pjoin
from utils.data_utils import encode

from plm import LightningPLM
from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel

from utils.model_utils import U_TKN, S_TKN
from dataloader import DELIMITER

repeatchars_pattern = re.compile('(\D)\\1{2,}')
doublespace_pattern = re.compile('\s+')

def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.log = getattr(args, 'log', True)

def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()

def proc_reply(reply):
    proc_text = re.sub('(<pad>|<unk>)', '', reply)
    return repeat_normalize(proc_text, num_repeats=4)

def tokenize(tokenizer, text, max_len):
    q_toked = tokenizer.tokenize(tokenizer.cls_token + text + tokenizer.sep_token)
    
    if len(q_toked) > max_len:
        q_toked = q_toked[:max_len-1] + [q_toked[-1]]

    token_ids = tokenizer.convert_tokens_to_ids(q_toked)
    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]

    return token_ids 

def eval_ar(args, model, tokenizer, device, test_data):
    u_tkn, s_tkn = U_TKN, S_TKN

    reply_list = []
    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']

            reply = ''
            q_toked = tokenizer.tokenize(u_tkn + query)
            if len(q_toked) >= args.max_len:
                q_toked = [q_toked[0]] + q_toked[-(int(args.max_len/2))+1:]

            for iter_ in range(args.max_len):
                r_toked = tokenizer.tokenize(s_tkn + reply)
                token_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(q_toked + r_toked)).to(device=device)

                logits = model(token_ids)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                if gen == DELIMITER:
                    gen = '#'

                reply += gen.replace('▁', ' ')

            reply= reply.strip()
            reply_list.append(reply)

            print("Reply: {}".format(reply))

        test_data['generated_reply'] = reply_list
        test_data.to_csv(f'{args.save_dir}/{args.model_name}.csv', index=False)
    

def eval_s2s(args, model, tokenizer, device, test_data):
    reply_list = []

    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            query = row['proc_query']

            enc_input, attention_mask = encode(tokenizer=tokenizer, \
                sent=tokenizer.bos_token+query+tokenizer.eos_token, \
                max_len=args.max_len)

            enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device=device)
            attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

            reply = ''
            for iter_ in range(args.max_len-1):
                dec_input, dec_attention_mask = encode(tokenizer=tokenizer, \
                    sent=tokenizer.bos_token+reply, max_len=args.max_len)

                dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device=device)
                dec_attention_mask = torch.FloatTensor(dec_attention_mask).unsqueeze(0).to(device=device)
    
                inputs = {
                    "input_ids": enc_input,
                    "attention_mask" : attention_mask,
                    "decoder_input_ids" : dec_input,
                    "decoder_attention_mask" : dec_attention_mask,
                    "labels": None
                }
                outs = model(inputs)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                if gen == DELIMITER:
                    gen = '#'
                reply += gen.replace('▁', ' ')

            reply= reply.strip()
            print("Query: {}".format(query))
            print("Reply: {}".format(reply))
            reply_list.append(reply)

        test_data['generated_reply'] = reply_list
        test_data.to_csv(f'{args.save_dir}/{args.model_name}.csv', index=False)
    
 
def eval_transformer(args, model, tokenizer, device, test_data):

    pred_list = []
    count = 0
    with torch.no_grad():
        for d in test_data.iterrows():
            row = d[1]
            text = row['proc_query']
            label = int(row['label'])

            assert isinstance(text, str)
            
            print("Input Text: %s" % text)

            input_ids = torch.LongTensor(tokenize(tokenizer, text=text, max_len=args.max_len)).unsqueeze(0).to(device=device)
            attention_mask = None

            logits = model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            print(predictions)
            pred_list.append(predictions[0]) 

            if predictions[0] == label:
                count += 1

        test_data['pred_reaction'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'{args.model_name}-{round(count/len(test_data), 2)*100}.csv'), sep='\t', index=False)
        print(f"Accuracy: {count/len(test_data)}")
              

def is_valid(query):
    if not re.sub('[\s]+', '', query):
        return False
    return True

def chat_ar(args, model, device):
    u_tkn, s_tkn = U_TKN, S_TKN
    tokenizer = model.tokenizer

    query = input('사용자 입력: ')
    
    with torch.no_grad():
        while is_valid(query):
            reply = ''
            q_toked = tokenizer.tokenize(u_tkn + query)
            if len(q_toked) >= args.max_len:
                query_toked = query_toked[-(int(args.max_len/2)):]

            for iter_ in range(args.max_len):
                a_toked = tokenizer.tokenize(s_tkn + reply)
                token_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(q_toked + a_toked)).to(device=device)

                logits = model(token_ids)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                reply += gen.replace('▁', ' ')

            reply= reply.strip()

            print("Query: {}".format(query))
            gen_replies = reply.split(DELIMITER)

            if len(gen_replies) < 2:
                print("Fail: {}".format(reply))
            else:
                print("Reply: {} (CLASS: {})".format(proc_reply(gen_replies[0]), gen_replies[-1]))
    
            query = input('사용자 입력: ')

        
def chat_s2s(args, model, device):
    tokenizer = model.tokenizer

    query = input('사용자 입력: ')
    with torch.no_grad():
        while is_valid(query):
            print("Query: {}".format(query))

            enc_input, attention_mask = encode(tokenizer=tokenizer, \
                sent=tokenizer.bos_token+query+tokenizer.eos_token, \
                max_len=args.max_len)

            enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device=device)
            attention_mask = torch.FloatTensor(attention_mask).unsqueeze(0).to(device=device)

            reply = ''
            for iter_ in range(args.max_len-1):
                dec_input, dec_attention_mask = encode(tokenizer=tokenizer, \
                    sent=tokenizer.bos_token+reply, max_len=args.max_len)

                dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device=device)
                dec_attention_mask = torch.FloatTensor(dec_attention_mask).unsqueeze(0).to(device=device)
    
                inputs = {
                    "input_ids": enc_input,
                    "attention_mask" : attention_mask,
                    "decoder_input_ids" : dec_input,
                    "decoder_attention_mask" : dec_attention_mask,
                    "labels": None
                }
                outs = model(inputs)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(outs.logits, dim=-1).squeeze().cpu().tolist())[-1]
                if gen == tokenizer.eos_token:
                    break
                if gen == DELIMITER:
                    gen = '#'
                reply += gen.replace('▁', ' ')

            reply= reply.strip()
            print("Reply: {}".format(reply))

            query = input('사용자 입력: ')

def evaluation(args, **kwargs):
    base_setting(args)
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    print(args.model_pt)
    if args.model_pt is not None:
        if args.model_type in ['bert', 'electra', 'bigbird', 'roberta']:
            model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
        elif args.model_type in ['gpt2']:
            model = AutoRegressiveModel.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))
        else:
            model = Seq2SeqModel.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args, device=torch.device(device))

    model = model.cuda()     
    model.eval()
    model.freeze()

    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    test_data.dropna(axis=0, inplace=True)

    if args.model_type in ['bert', 'electra', 'bigbird', 'roberta']:
        eval_transformer(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)

    elif args.model_type in ['gpt2']:
        if args.chat:
            chat_ar(args, model, device)
        else:
            eval_ar(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)
    else:
        if args.chat:
            chat_s2s(args, model, device)
        else:
            eval_s2s(args, model=model, tokenizer=model.tokenizer, device=device, test_data=test_data)

            

