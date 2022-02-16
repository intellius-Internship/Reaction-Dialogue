from transformers import (GPT2LMHeadModel, 
                        PreTrainedTokenizerFast,
                        BartForConditionalGeneration)

from kobart import get_kobart_tokenizer

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

skt_kobart = './cache/kobart_from_pretrained'
def load_model(model_name, cache_dir='./cache'):
    if 'bart' == model_name:
        # model = BartForConditionalGeneration.from_pretrained(skt_kobart)
        # tokenizer = get_kobart_tokenizer(cache_dir)
        
        model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        return model, tokenizer

    elif 'gpt2' == model_name:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
        return model, tokenizer


    raise NotImplementedError('Unknown model')

def load_tokenizer(model_name, cache_dir='./cache'):
    if 'bart' == model_name:
        return get_kobart_tokenizer(cache_dir)

    elif 'gpt2' == model_name:
        return PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

    raise NotImplementedError('Unknown model')