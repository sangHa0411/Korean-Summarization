import wandb

import os
import random
import numpy as np

import torch

from dotenv import load_dotenv
from metric import compute_metrics
from preprocessor import tokenize_function

import argparse
from functools import partial
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train(args):

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Data
    dataset_name = "sh110495/book-summarization"
    HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")    
    dataset = load_dataset(dataset_name, use_auth_token=HUGGINGFACE_KEY)
    print(dataset)

    # -- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM, use_fast=True)

    # -- Preprocessing
    print('\nTokenizing Data')

    prep_fn = partial(tokenize_function, tokenizer=tokenizer, max_input_length=args.max_input_len, max_target_length=args.max_target_len)
    tokenized_dataset = dataset.map(prep_fn, 
        batched=True, 
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset['train'].column_names,
        load_from_cache_file=True
    )

    train_data = tokenized_dataset['train'] # train data
    val_data = tokenized_dataset['validation'] # validation data

    # -- Config
    config = AutoConfig.from_pretrained(args.PLM)
    config.max_length = 128
    config.num_beams = 1

    # -- Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.PLM, config=config).to(device)

    # -- Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -- Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,                                   # output directory
        logging_dir = args.logging_dir,                                 # logging directory
        num_train_epochs = args.epochs,                                 # epochs
        save_steps = args.eval_steps,                                   # model saving steps
        eval_steps = args.eval_steps,                                   # evaluation steps
        logging_steps = args.log_steps,                                 # logging steps
        evaluation_strategy = args.evaluation_strategy,                 # evaluation strategy
        per_device_train_batch_size = args.train_batch_size,            # train batch size
        per_device_eval_batch_size = args.eval_batch_size,              # evaluation batch size
        warmup_steps=args.warmup_steps,                                 # warmup steps
        weight_decay=args.weight_decay,                                 # weight decay
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # gradient accumulation steps
        eval_accumulation_steps=args.gradient_accumulation_steps,       # eval_accumulation_steps
        overwrite_output_dir=True,
        predict_with_generate=True
    )
    
    # -- Metrics
    metric_fn = partial(compute_metrics, tokenizer=tokenizer)
  
    # -- Trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_fn
    )

    # -- Training
    print('\nTraining')
    trainer.train()
    
def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = "base-model"
    wandb.init(
        entity="sangha0411",
        project="Abstractive-Summarization", 
        name=wandb_name,
        group='kobart')

    wandb.config.update(args)
    train(args)
    wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Training Argument
    parser.add_argument('--output_dir', default='exps', help='model save directory')
    parser.add_argument('--logging_dir', default='logs', help='logging directory')
    parser.add_argument('--PLM', type=str, default='gogamza/kobart-base-v2', help='model type (default: gogamza/kobart-base-v2)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='eval batch size (default: 32)')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='number of warmup steps for learning rate scheduler (default: 5000)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='strength of weight decay (default: 1e-4)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='evaluation strategy to adopt during training, steps or epoch (default: steps)')
    parser.add_argument('--eval_steps', type=int, default=5000, help='evaluation steps (5000)')
    parser.add_argument('--log_steps', type=int, default=1000, help='evaluation steps (1000)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps (defalut: 1)')
    parser.add_argument('--max_input_len', type=int, default=1024, help='max length of input tensor (default: 1024)')
    parser.add_argument('--max_target_len', type=int, default=64, help='max length of target tensor (default: 64)')
    parser.add_argument('--preprocessing_num_workers', type=int, default=4, help='preprocessing num workers (default: 4)')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='path.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

