#!/bin/bash
learning=(5e-5 3e-5 2e-5)
gpu=$1
label=$2
input=$3
sent1=$4
sent2=$5
bert=$6

for lr in "${learning[@]}"
do
    echo "python -m attribute_clf.clf_sentence_pairs --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input ${input} --output attribute_clf/cross_validation_${label}  --sent1 ${sent1} --sent2 ${sent2} --bert_model ${bert}"
    python -m attribute_clf.clf_sentence_pairs --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input "${input}" --output "attribute_clf/cross_validation_${label}" --sent1 ${sent1} --sent2 ${sent2} --bert_model "${bert}"
done
