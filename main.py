import torch
import argparse
import logging
import random
import numpy as np

import warnings
import transformers
import pytorch_lightning as pl

from auto_regressive_model import AutoRegressiveModel
from seq2seq_model import Seq2SeqModel
from eval import evaluation

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 19

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generative chatbot based on Auto-Regressive Model')
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='for training')

    parser.add_argument('--chat',
                        action='store_true',
                        default=False)

    parser.add_argument('--max_len',
                        type=int,
                        default=128)

    parser.add_argument('--data_dir',
                        type=str,
                        default='data')
                        
    parser.add_argument('--save_dir',
                        type=str,
                        default='result')

    parser.add_argument('--model_name',
                        type=str,
                        default='kogpt_chat')

    parser.add_argument('--model_type',
                        type=str,
                        default='gpt2')

    parser.add_argument('--model_pt',
                        type=str,
                        default=None)

    parser.add_argument("--gpuid", nargs='+', type=int, default=0)

    parser = AutoRegressiveModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    set_seed(SEED)

    global DATA_DIR
    DATA_DIR = args.data_dir

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_ckpt',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix=f'{args.model_name}'
        )
        # python main.py --train --gpuid 0 1 2 --max_epochs 5 --data_dir data --model_name gpt_chat --model_type gpt2
        model = AutoRegressiveModel(args) if args.model_type == 'gpt2' else Seq2SeqModel(args)
        model.train()
        trainer = Trainer(
                        check_val_every_n_epoch=1, 
                        checkpoint_callback=checkpoint_callback, 
                        flush_logs_every_n_steps=100, 
                        gpus=args.gpuid, 
                        gradient_clip_val=1.0, 
                        log_every_n_steps=50, 
                        logger=True, 
                        max_epochs=args.max_epochs, 
                        num_processes=1,
                        accelerator='ddp')
        
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    else:
        with torch.cuda.device(args.gpuid[0]):
            evaluation(args)