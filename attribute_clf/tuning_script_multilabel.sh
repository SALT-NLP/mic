#!/bin/bash
learning=(5e-5 3e-5 2e-5)
gpu=$1
label=$2
input=$3
sent1=$4
bert=$5

for lr in "${learning[@]}"
do
    echo "python -m attribute_clf.clf_sentence_pairs_multilabel --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input ${input} --output attribute_clf/cross_validation_${label}  --sent1 ${sent1} --sent2 ${sent2} --label_classes 0 1 2 3 4 5 --bert_model ${bert}" 
    python -m attribute_clf.clf_sentence_pairs_multilabel --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input "${input}" --output "attribute_clf/cross_validation_${label}" --sent1 ${sent1} --label_classes 0 1 2 3 4 5 --bert_model "${bert}"
    echo "python -m attribute_clf.clf_sentence_pairs_multilabel --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input ${input} --output attribute_clf/cross_validation_backtr_${label}  --sent1 ${sent1} --sent2 ${sent2} --label_classes 0 1 2 3 4 5 --bert_model ${bert} --backtranslate attribute_clf/backtranslations/mixed_en_de_ru_backtranslations.pkl" 
    python -m attribute_clf.clf_sentence_pairs_multilabel --batchsize 16 --epochs 8 --lr ${lr} --gpu ${gpu} --label ${label} --input "${input}" --output "attribute_clf/cross_validation_backtr_${label}" --sent1 ${sent1} --label_classes 0 1 2 3 4 5 --bert_model "${bert}" --backtranslate "attribute_clf/backtranslations/mixed_en_de_ru_backtranslations.pkl"
done