#!/bin/bash
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

learning=(5e-5 3e-5 2e-5)
gpu=$1
label=$2
input=$3

for lr in "${learning[@]}"
do
    echo "python qa_filtering_clf/sentence_pairs_clf.py --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input ${input} --output qa_filtering_clf/cross_validation_${label}"
    python qa_filtering_clf/sentence_pairs_clf.py --batchsize 16 --epochs 8 --lr "${lr}" --gpu "${gpu}" --label "${label}" --input "${input}" --output "qa_filtering_clf/cross_validation_${label}"
done
