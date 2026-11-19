import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time
import csv
from model import *
from generate import generate
from torch.utils import data
from torch.utils.data import DataLoader
from qa_dataset import qa_dataset,build_vocab
import pickle


# TODO (可选作业：Attention 对齐的 Seq2Seq 问答)
# 要求：阅读文献，手动「古法」实现注意力机制（不要调用现成的 Attention 代码，否则不得分）。
# 建议：
#   1) 复用 Q2 的编码器（model.py 里 QA_RNN 的 embed + encoder 部分）；
#   2) 在本文件里单独实现一个「注意力对齐解码器」AttnDecoderRNN：
#      - 需手动完成 score -> softmax(对齐权重) -> context(加权求和) 三步；
#      可参考 Bahdanau et al. 2015 (additive) 或 Luong et al. 2015 (dot/general)。
#   3) 本题代码全部整合在 train_attention.py 这一个文件里（数据集等仍复用现有模块）。

# 代码各个部分请手动添加详细注释。

def main(args):
    pass


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filename', type=str)
    argparser.add_argument('--model', type=str, default="gru", choices=['gru','lstm'])
    argparser.add_argument('--n_epochs', type=int, default=100)
    argparser.add_argument('--print_every', type=int, default=1)
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.006)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--cuda', action='store_true')
    
    # 建议加载已训练好的编码器模型
    argparser.add_argument('-m_enc', '--encoder_model_path', type=str, default='saves/rnn.pt')

    args = argparser.parse_args()

    main(args)
