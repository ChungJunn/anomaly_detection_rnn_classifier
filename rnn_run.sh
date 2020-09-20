DATA='cnsm_exp2_1_data'
DATA_DIR=$HOME'/autoregressor/data/'$DATA

if [ $DATA = 'cnsm_exp1_data' ]
then
    DIM_INPUT=110
else
    DIM_INPUT=88
fi

RNN_LEN=16 # this means 17 in supervised learning case
BATCH_SIZE=32
OPTIMIZER='RMSprop'
LR=0.01
MAX_EPOCH=20
VALID_EVERY=1
PATIENCE=1
DIM_LSTM_HIDDEN=64
DIM_FC_HIDDEN=32
DIM_OUT=2
NAME=$DATA
TAG='none'

TR_PATH=$DATA_DIR'/suprnn_train.rnn_len'$RNN_LEN'.pkl'
VAL_PATH=$DATA_DIR'/suprnn_val.rnn_len'$RNN_LEN'.pkl'
TEST_PATH=$DATA_DIR'/suprnn_test.rnn_len'$RNN_LEN'.pkl'
STAT_FILE=$DATA_DIR'/ar_train.rnn_len'$RNN_LEN'.pkl.stat'

python3 rnn_main.py \
    --tr_path=$TR_PATH \
    --val_path=$VAL_PATH \
    --test_path=$TEST_PATH \
    --stat_file=$STAT_FILE \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --optimizer=$OPTIMIZER \
    --max_epoch=$MAX_EPOCH \
    --valid_every=$VALID_EVERY \
    --patience=$PATIENCE \
    --dim_input=$DIM_INPUT \
    --dim_out=$DIM_OUT \
    --dim_fc_hidden=$DIM_FC_HIDDEN \
    --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
    --rnn_len=$RNN_LEN \
    --name=$NAME \
    --tag=$TAG 

