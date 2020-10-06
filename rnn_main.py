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

from rnn_model import RNN_MODEL1, RNN_ESM
from rnn_data import RNNIterator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''
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

'''
def test(model, test_loader, device):
    preds = []
    targets = []
    
    with torch.no_grad():
        for xs, ys, end_of_data in test_loader:
            xs, ys = xs.to(device), ys.to(device)
            output = model(xs) # output: rnn_len x bsz x 2 
            output = output[-1,:,:] # bsz x 2 

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

def evaluate_(model, validiter, criterion, device):
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

def train(
    net: RNN_MODEL1,
    train_loader: RNNIterator,
    valid_loader: RNNIterator,
    patience: int,
    args: object,
    dtype: torch.dtype,
    device: torch.device,
    savedir: str,
    neptune: neptune,
) -> RNN_MODEL1:
    """
    Train CNN on provided data set.
    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(device)  # pyre-ignore [28]
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    mystring = "optim." + args.optimizer

    if args.optimizer == 'Adam':
        optimizer = eval(mystring)(net.parameters(), lr=args.lr)
    else:
        optimizer = eval(mystring)(net.parameters(), lr=args.lr)

    print("=" * 90)
    print('data_dim: {:d} | data_len {:d}'.format(train_loader.in_n, train_loader.data_len))
    print('data_dim: {:d} | data_len {:d}'.format(valid_loader.in_n, valid_loader.data_len))
    print("=" * 90)
    print(net)
    print("=" * 90)
    print(optimizer)
    print("=" * 90)

    num_epochs = 1000
    bc = 0
    best_val = 0
    best_net = None
    train_loss=0.0

    # Train Network
    # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
    for epoch in range(num_epochs):
        if epoch >= args.max_epoch:
            break

        for iloop, (inputs, labels, end_of_data) in enumerate(train_loader):
            net.train()
            if end_of_data == 1:
                break
            
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(dtype=torch.int64, device=device).squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)[-1,:,:] # for RNN
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
        
        # check if validation improves
        train_loss = train_loss / iloop
        val = evaluate_(net, valid_loader, criterion, device)
        print('epoch: {:d} | train: {:.4f} | val: {:.4f}'.format(epoch+1, train_loss, val))
        #neptune.log_metric('tr loss', epoch, train_loss)
        #neptune.log_metric('val loss', epoch, val)
        if epoch==0 or val < best_val:
            # if yes, save model, init bc, and best_val
            torch.save(net, savedir)
            bc = 0
            best_val = val
        else:
            # if no, bc++, check if patience is over
            bc += 1
            if bc > patience:
                break


    print('training over')
    best_net = torch.load(savedir)
    return best_net

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

    if args.n_cmt > 1:
        model = RNN_ESM(
                n_cmt = args.n_cmt,
                dim_input = args.dim_input,
                dim_lstm_hidden=args.dim_lstm_hidden,
                dim_fc_hidden=args.dim_fc_hidden,
                dim_output=args.dim_out).to(device)
    elif args.n_cmt == 1:
        model = RNN_MODEL1(
                dim_input = args.dim_input,
                dim_lstm_hidden=args.dim_lstm_hidden,
                dim_fc_hidden=args.dim_fc_hidden,
                dim_output=args.dim_out).to(device)
    else:
        print('n_cmt must be natual number')
        import sys; sys.exit(0)

    start = time.time()

    # train the model
    if args.n_cmt > 1:
        for i in range(args.n_cmt):
            model.model_list[i] = train(
                net=model.model_list[i],
                train_loader= trainiter,
                valid_loader= validiter,
                patience=args.patience,
                args=args,
                dtype=torch.float32,
                device=device,
                savedir=args.out_dir + '/' + args.out_file,
                neptune=neptune)
    else:
        model = train(
            net=model,
            train_loader= trainiter,
            valid_loader= validiter,
            patience=args.patience,
            args=args,
            dtype=torch.float32,
            device=device,
            savedir=args.out_dir + '/' + args.out_file,
            neptune=neptune)

    acc, prec, rec, f1 = test(model, testiter, device)
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))
    
    neptune.set_property('acc', acc)
    neptune.set_property('prec', prec)
    neptune.set_property('rec', rec)
    neptune.set_property('f1', f1)

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
    parser.add_argument('--n_cmt', type=int, help='')
    args = parser.parse_args()

    params = vars(args)

    neptune.init('cjlee/AnomalyDetection-Supervised-RNN')
    experiment = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)

    args.out_dir='./result'
    args.out_file=experiment.id + '.pth'

    # temporary code for testing
    train_main(args, neptune)
