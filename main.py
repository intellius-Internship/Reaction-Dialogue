import random
import torch
import argparse
import logging
import warnings

import numpy as np
import transformers

from plm import LightningPLM
from eval import evaluation

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SEED = 19

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reply Classification based on PLM')
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='for training')

    parser.add_argument('--data_dir',
                        type=str,
                        default='data')

    parser.add_argument('--save_dir',
                        type=str,
                        default='result')

    parser.add_argument('--model_name',
                        type=str,
                        default='baseline')

    parser.add_argument('--model_type',
                        type=str,
                        default='bert')
                        
    parser.add_argument('--num_labels',
                        type=int,
                        default=115)

    parser.add_argument('--model_pt',
                        type=str,
                        default='baseline-last.ckpt')

    parser.add_argument('--monitor',
                        type=str,
                        default='val_loss')
            
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)

    parser = LightningPLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    global DATA_DIR
    DATA_DIR = args.data_dir

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_ckpt',
            filename='{epoch:02d}-{avg_val_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='avg_val_loss',
            mode='min',
            prefix=f'{args.model_name}'
        )
        # python main.py --train --gpuid 0 1 2 --max_epochs 5 --data_dir data --model_name bert_chat --model_type bert
        model = LightningPLM(args)
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
                        accelerator='ddp' if args.model_type in ['bert', 'electra'] else None)
        
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))

    else:
        with torch.cuda.device(args.gpuid[0]):
            evaluation(args)