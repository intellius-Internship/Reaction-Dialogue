from transformers import (ElectraForSequenceClassification, ElectraTokenizer, ElectraConfig,
                        BertForSequenceClassification, BertConfig,
                        AutoModelForSequenceClassification, AutoTokenizer, AutoConfig)

from tokenization import KoBertTokenizer

def load_model(model_type, num_labels, labels=None, cache_dir='./cache'):
    if labels is None:
        labels = list(range(num_labels))

    if 'bert' == model_type:
        config = BertConfig.from_pretrained(
            "monologg/kobert",
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)}
        )
    
        model = BertForSequenceClassification.from_pretrained(
            "monologg/kobert", 
            config=config
        )
        tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        return model, tokenizer

    elif 'electra' == model_type:
        config = ElectraConfig.from_pretrained(
            "monologg/koelectra-base-v3-discriminator",
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)}
        )
        model = ElectraForSequenceClassification.from_pretrained(
            "monologg/koelectra-base-v3-discriminator", 
            config=config
        )
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        return model, tokenizer

    elif 'bigbird' == model_type:
        config = AutoConfig.from_pretrained("monologg/kobigbird-bert-base", 
                num_labels=num_labels,
                cache_dir=cache_dir)
        config.label2id = {str(i): label for i, label in enumerate(labels)}
        config.id2label = {label: i for i, label in enumerate(labels)}
        model = AutoModelForSequenceClassification.from_pretrained(
            "monologg/kobigbird-bert-base", 
            config=config
        )
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        return model, tokenizer

  
    raise NotImplementedError('Unknown model')