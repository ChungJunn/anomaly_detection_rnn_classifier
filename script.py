import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='')
args = parser.parse_args()

datasets = ['cnsm_exp1_data', 'cnsm_exp2_1_data', 'cnsm_exp2_2_data']

n_exp = 5

for dataset in datasets:
    for i in range(n_exp):
        cmd = 'rnn_run.sh ' + str(args.gpu) + ' ' + dataset
        os.system(cmd)
