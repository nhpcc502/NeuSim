#!/bin/bash

models=(
    gru
    # bi_gru
    lstm
    bert
)

batch_size=32
epochs=1000
train=0

for model in ${models[@]}
do
    echo "***************************************************************"
    echo "***************************************************************"
    time=$(date "+%d%H%M")
    python -u seq2seq.py --models=$model --dataset=np_mix --batch_size=$batch_size --train=$train --epochs=$epochs | tee result_$model.log
    echo
    echo
done

echo "***************************************************************"
echo "***************************************************************"
time=$(date "+%d-%H-%M")
python -u graph2seq.py --dataset=np_mix --batch_size=$batch_size --epochs=$epochs | tee logs/graphmr_$dataset.log
echo
echo
