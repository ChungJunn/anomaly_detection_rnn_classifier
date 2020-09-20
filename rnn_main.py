import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import os
import math
import sys
import time

import argparse
import neptune

from rnn_model import RNN_MODEL1
from rnn_data import RNNIterator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, input, target, optimizer, criterion, device):
    model.train()

    hidden, cell = model.init_hidden(input.shape[1])
    hidden, cell = Variable(hidden).to(device), Variable(cell).to(device)

    optimizer.zero_grad()
    output = model(input, (hidden, cell))

    loss = criterion(output[-1,:,:], target.type(torch.int64).squeeze()) # reduce to scalar?
    
    loss.backward()
    optimizer.step()

    return output, loss.item()

def evaluate(model, validiter, criterion, device, args):
    model.eval()
    total_loss = 0
    n_segs = 0

    for xs, ys, end_of_data in validiter:
        xs, ys = xs.to(device), ys.to(device)

        outs = model(xs)

        loss = criterion(outs[-1,:,:], ys.type(torch.int64).squeeze())

        total_loss += loss.item()
        n_segs += 1
        
        if end_of_data == 1: break
        
    return total_loss / n_segs

def test(model, test_loader, device):
    correct = 0
    total = 0

    preds = []
    targets = []
    
    with torch.no_grad():
        for xs, ys, end_of_data in test_loader:
            xs, ys = xs.to(device), ys.to(device)
            output = model(xs)
            output = output[-1,:,:]

            _, output_index = torch.max(output,1)
            output_index = output_index.reshape(-1,1).detach().cpu().numpy()
            ys = ys.cpu().numpy()

            preds.append(output_index)
            targets.append(ys)

            if end_of_data == 1: break
    
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return acc, prec, rec, f1

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_main(args, neptune):
    device = torch.device("cuda")

    # iterators
    trainiter = RNNIterator(args.tr_path, stat_file=args.stat_file, batch_size = args.batch_size)
    validiter = RNNIterator(args.val_path, stat_file=args.stat_file, batch_size = args.batch_size)
    testiter = RNNIterator(args.test_path, stat_file=args.stat_file, batch_size = args.batch_size)

    # model
    model = RNN_MODEL1(
            dim_input = args.dim_input,
            dim_lstm_hidden=args.dim_lstm_hidden,
            dim_fc_hidden=args.dim_fc_hidden,
            dim_output=args.dim_out).to(device)

    # training parameters 
    mystring = "optim." + args.optimizer
    optimizer = eval(mystring)(model.parameters(), args.lr)
    criterion = F.nll_loss

    start = time.time()

    loss_total = 0
    epoch=0
   
    for iloop, (tr_x, tr_y, end_of_data) in enumerate(trainiter):
        tr_x, tr_y = Variable(tr_x).to(device), Variable(tr_y).to(device)
        output, loss = train(model, tr_x, tr_y, optimizer, criterion, device)
        loss_total += loss
        
        if end_of_data == 1:
            epoch += 1
            print("%d (%s) %.4f" % (epoch, timeSince(start), loss_total / (trainiter.data_len / args.batch_size)))
            neptune.log_metric('train loss', epoch, loss_total / (trainiter.data_len / args.batch_size))
            torch.save(model, args.out_dir + '/' + args.out_file)
            loss_total=0

            val_loss = evaluate(model, validiter, criterion, device, args)
            print('val_loss: {:f}'.format(val_loss))
            neptune.log_metric('epoch/valid loss', epoch, val_loss)

            if epoch >= args.max_epoch: break

    acc, prec, rec, f1 = test(model, testiter, device)
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', type=str, help='')
    parser.add_argument('--val_path', type=str, help='')
    parser.add_argument('--test_path', type=str, help='')
    parser.add_argument('--stat_file', type=str, help='')

    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--lr', type=float, help='')
    parser.add_argument('--optimizer', type=str, help='')
    parser.add_argument('--max_epoch', type=int, help='')
    parser.add_argument('--valid_every', type=int, help='')
    parser.add_argument('--out_dir', type=str, help='')
    parser.add_argument('--out_file', type=str, help='')
    parser.add_argument('--patience', type=int, help='')
    parser.add_argument('--is_train', type=int, help='')

    parser.add_argument('--dim_input', type=int, help='')
    parser.add_argument('--dim_out', type=int, help='')
    parser.add_argument('--dim_lstm_hidden', type=int, help='')
    parser.add_argument('--dim_fc_hidden', type=int, help='')

    parser.add_argument('--rnn_len', type=int, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--tag', type=str, help='')
    args = parser.parse_args()

    params = vars(args)

    neptune.init('cjlee/AnomalyDetection-Supervised-RNN')
    experiment = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)

    args.out_dir='./result'
    args.out_file=experiment.id + '.pth'

    # temporary code for testing
    train_main(args, neptune)
