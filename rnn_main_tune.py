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

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


hparam_to_be_tuned = ['lr', 'momentum', 'weight_decay', 'dim_lstm_hidden', 'dim_fc_hidden']


def test(model, test_loader, device):
    preds = []
    targets = []
    
    model.eval()
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

def train_test_main(config, args):
    #Apply target hyper pramter values
    for hp in hparam_to_be_tuned:
        setattr(args, hp, config[hp])


    device = torch.device("cuda")

    # iterators
    trainiter = RNNIterator(args.tr_path, stat_file=args.stat_file, batch_size = args.batch_size)
    validiter = RNNIterator(args.val_path, stat_file=args.stat_file, batch_size = args.batch_size)
    testiter = RNNIterator(args.test_path, stat_file=args.stat_file, batch_size = args.batch_size)

    model = RNN_MODEL1(
            dim_input = args.dim_input,
            dim_lstm_hidden=args.dim_lstm_hidden,
            dim_fc_hidden=args.dim_fc_hidden,
            dim_output=args.dim_out).to(device)

    start = time.time()

    model = train(
        net=model,
        train_loader= trainiter,
        valid_loader= validiter,
        patience=args.patience,
        args=args,
        dtype=torch.float32,
        device=device,
        savedir=None,
        neptune=neptune)
    
    acc, prec, rec, f1 = test(model, test_loader, device)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}




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
):
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

    trial_name = tune.get_trial_name()
    print(trial_name, "Train")
    if args.optimizer == 'Adam':
        print("Adam optimizer")
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
#        optimizer = eval(mystring)(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        print("sgd optimizer")
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    elif args.optimizer == 'RMSprop':
        print("RMSprop optimizer")
        optimizer = optim.RMSprop(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    else:
        print("Optimizer Exception. args.optimizer:", args.optimizer)
        raise Exception()
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
    print("Training start" , num_epochs)
    for epoch in range(num_epochs):
        if epoch >= args.max_epoch:
            break

        net.train()
        for iloop, (inputs, labels, end_of_data) in enumerate(train_loader):
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
        print("Train end", iloop)
        train_loss = train_loss / iloop
        print("Train loss:", train_loss)


        print("Test start")
        acc, prec, rec, f1 = test(net, valid_loader, device)

        tune.report(acc = acc, prec = prec, rec = rec, f1 = f1)
        #val = evaluate_(net, valid_loader, criterion, device)
        #print('epoch: {:d} | train: {:.4f} | val: {:.4f}'.format(epoch+1, train_loss, val))
        #neptune.log_metric('tr loss', epoch, train_loss)
        #neptune.log_metric('val loss', epoch, val)
        #if epoch==0 or val < best_val:
            # if yes, save model, init bc, and best_val
        #    torch.save(net, savedir)
        #    bc = 0
        #    best_val = val
        #else:
            # if no, bc++, check if patience is over
        #    bc += 1
        #    if bc > patience:
        #        break


    #print('training over')
    #best_net = torch.load(savedir)
    #return best_net

    return

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_main(config, checkpoint_dir = None, args = None, neptune = None):
    #Apply target hyper pramter values
    for hp in hparam_to_be_tuned:
        setattr(args, hp, config[hp])


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

    model = train(
        net=model,
        train_loader= trainiter,
        valid_loader= validiter,
        patience=args.patience,
        args=args,
        dtype=torch.float32,
        device=device,
        savedir=None,
        neptune=neptune)

    #acc, prec, rec, f1 = test(model, testiter, device)
    

def timestamp_expr_id():
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H_%M_%S_%f")[:-3]

    return formatted_time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', type=str, help='')
    parser.add_argument('--val_path', type=str, help='')
    parser.add_argument('--test_path', type=str, help='')
    parser.add_argument('--stat_file', type=str, help='')

    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--lr', type=float, help='')
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
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
    parser.add_argument('--htune', type=str, default='random')
    args = parser.parse_args()

    params = vars(args)

    exprid = timestamp_expr_id()

    args.n_cmt = 1

    tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 3000000
    if args.htune == 'random':
        config = {
                'lr' : tune.quniform(1e-4, 1, 1e-4),
                'momentum' : tune.quniform(0.8, 0.99, 0.01),
                'dim_lstm_hidden' : tune.choice([128, 256, 512]) ,
                'dim_fc_hidden' : tune.sample_from(lambda spec: spec.config.dim_lstm_hidden // (2 ** (np.random.randint(0, 3))) ),
                'weight_decay' : tune.choice([1e-2, 1e-3, 1e-4, 1e-5])
                }

        asha_scheduler = ASHAScheduler(
            metric = 'f1',
            mode = 'max',
            grace_period = 10,
                )

        reporter = CLIReporter( metric_columns = ["f1", "acc", "prec", "rec"] )
        ray_result = tune.run(
                partial(train_main, args = args, neptune = None),
                name = exprid,
                config = config,
                local_dir = 'ray_results',
                resources_per_trial = {'cpu': 1, 'gpu': 0.2},
                num_samples = 500,
                scheduler = asha_scheduler,
                fail_fast = True,
                progress_reporter = reporter,
                log_to_file = True)



    #neptune.init('cjlee/AnomalyDetection-Supervised-RNN')
    #experiment = neptune.create_experiment(name=args.name, params=params)
    #neptune.append_tag(args.tag)

    #args.out_dir='./result'
    #args.out_file=experiment.id + '.pth'

    # temporary code for testing
    #train_main(args, neptune)


    best_trial = ray_result.get_best_trial("f1", "max", "all")
    best_config = best_trial.config
    print("Best tiral config: {}".format(best_config))
    print("Best tiral final f1 score: {}".format(best_trial.last_result["f1"]))

    test_num = 5

    print("Test")
    avr_f1_score = 0
    for i in range(test_num):
        test_result = train_test_main(best_config, args)

        print("Test{} result: {}".format(i+1, test_result))
        avr_f1_score += test_result['f1']

    avr_f1_score /= test_num
    print("Average f1 score from total {} tests: {}".format(test_num, avr_f1_score))



