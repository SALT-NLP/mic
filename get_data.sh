#!/bin/bash
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

mkdir resources
mkdir data/prepared
mkdir -p data/raw/Reddit-QA-Corpus
cd resources || exit
wget https://github.com/aruljohn/popular-baby-names/raw/master/2018/boy_names_2018.csv
wget https://github.com/aruljohn/popular-baby-names/raw/master/2018/girl_names_2018.csv
cd ../data/raw/Reddit-QA-Corpus || exit
wget http://files.fionn.xyz/Datasets/Reddit/Answers_R.txt
wget http://files.fionn.xyz/Datasets/Reddit/Questions_R.txt
wget https://databank.illinois.edu/datafiles/pjwpj/download -O Enhanced_Morality_Lexicon_V1.1.txt
