import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer
from data_loader import load_and_cache_examples

import numpy as np

import torch

def main(args,args2):
    init_logger()  # 记录器

    tokenizer = load_tokenizer(args)  # 下载vocabulary和cache

    # 六维 1、all_input_ids 2、all_attention_mask 3、all_token_type_ids 4、all_label_ids 5、all_e1_mask 6、all_e2_mask
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    #dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    # 加载训练器，配置参数来实例化模型
    #trainer = Trainer(args, args2,train_dataset=train_dataset, test_dataset=test_dataset, dev_dataset=dev_dataset)
    trainer = Trainer(args,  args2,train_dataset=train_dataset, test_dataset=test_dataset)


    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('test')
    trainer.train()
    trainer.load_model()
    trainer.evaluate('test')


if __name__ == '__main__':
    # 定义解析器
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ppi", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument("--pretrained_model_name", default="monologg/biobert_v1.1_pubmed", required=False,
                        help="Pretrained model name")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # 修改过原来16
    #parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training and evaluation.")
    # 修改过原来300
    # parser.add_argument("--max_seq_len", default=400, type=int,
    #                     help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_len", default=400, type=int,
                        help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")


    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    #parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay if we apply some.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    # 原来0.1
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

# 原来400
    parser.add_argument('--logging_steps', type=int, default=400, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=400, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()


    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser2.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser2.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser2.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser2.add_argument('--lr', type=float, default=2e-5,
                        help='Initial learning rate.')
    parser2.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser2.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser2.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')

    args2 = parser2.parse_args()
    args2.cuda = not args2.no_cuda and torch.cuda.is_available()

    np.random.seed(args2.seed)
    torch.manual_seed(args2.seed)
    if args2.cuda:
        torch.cuda.manual_seed(args2.seed)
    main(args,args2)
