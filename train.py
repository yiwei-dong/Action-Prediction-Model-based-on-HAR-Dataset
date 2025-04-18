import logging
import argparse
from DataProcess import *
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from forecasting_model import *
from train_model import *
import os

def run(args):
    print(args)
    if args.use_gpu:
        device = torch.device(f'cuda:{args.gpu_num}')
    else:
        device = torch.device("cpu")

    train_data_path = os.path.join(args.data_dir, 'train')
    test_data_path = os.path.join(args.data_dir, 'test')
    train_dataloader = DataProcess('train', train_data_path, args.use_gpu, args.gpu_num).get_loader(args.batch_size,args.num_workers)
    test_dataloader = DataProcess('test', test_data_path, args.use_gpu, args.gpu_num).get_loader(args.batch_size,args.num_workers)

    prediction_model = HARTransformer(input_dimension=561, num_layers=6, dimension_model=args.dimension_model,
                                      nhead=args.nhead, dimension_feedforward=args.dim_feedforward, dropout=args.dropout).to(device)
    training_model(args, prediction_model, train_dataloader, test_dataloader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='action prediction')
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--log_dir', type=str, default='./tf-logs/')
    parser.add_argument('--save_dir', type=str, default='./best_model/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dimension_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--dim_feedforward', type=int, default=256)

    args = parser.parse_args()
    run(args)