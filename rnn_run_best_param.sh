DATA=$2 #'cnsm_exp2_1_data'
DATA_DIR=$HOME'/autoregressor/data/'$DATA

if [ $DATA = 'cnsm_exp1_data' ]
then
    DIM_INPUT=110
else
    DIM_INPUT=88
fi

RNN_LEN=16 # this means 17 in supervised learning case
BATCH_SIZE=256 #Diff
OPTIMIZER='SGD'
#OPTIMIZER='RMSprop'
LR=0.2017 #Diff sgd
#LR=0.0002 #Diff RMSprop
MAX_EPOCH=200 #Diff
VALID_EVERY=1
PATIENCE=20 #Diff
DIM_LSTM_HIDDEN=512 #Diff sgd
#DIM_LSTM_HIDDEN=256 #Diff RMSprop
DIM_FC_HIDDEN=128 #Diff sgd
#DIM_FC_HIDDEN=64 #Diff RMSprop
MOMENTUM=0.8 #Diff sgd
#MOMENTUM=0.89 #Diff RMSprop
WEIGHT_DECAY=1e-05 #Diff RMSprop sgd
DIM_OUT=2
NAME=$DATA'-ensemble'
TAG='none'
N_CMT=1

TR_PATH=$DATA_DIR'/suprnn_train.rnn_len'$RNN_LEN'.pkl'
VAL_PATH=$DATA_DIR'/suprnn_val.rnn_len'$RNN_LEN'.pkl'
TEST_PATH=$DATA_DIR'/suprnn_test.rnn_len'$RNN_LEN'.pkl'
STAT_FILE=$DATA_DIR'/ar_train.rnn_len'$RNN_LEN'.pkl.stat'

export CUDA_VISIBLE_DEVICES=$1

python3 rnn_main_copy.py \
    --tr_path=$TR_PATH \
    --val_path=$VAL_PATH \
    --test_path=$TEST_PATH \
    --stat_file=$STAT_FILE \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --max_epoch=$MAX_EPOCH \
    --valid_every=$VALID_EVERY \
    --patience=$PATIENCE \
    --dim_input=$DIM_INPUT \
    --dim_out=$DIM_OUT \
    --dim_fc_hidden=$DIM_FC_HIDDEN \
    --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
    --rnn_len=$RNN_LEN \
    --name=$NAME \
    --tag=$TAG \
    --n_cmt=$N_CMT 
