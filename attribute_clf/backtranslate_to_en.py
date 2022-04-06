import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import re
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='clean_moral_eval_aug21.csv')
parser.add_argument('--target', type=str, default='rot')
parser.add_argument('--label', type=str, default='violation-severity')
parser.add_argument('--skiplabel', type=float, default=None, help='label to skip when backtranslating (the majority class)')
parser.add_argument('--temperature', type=float, default = 0.9)
parser.add_argument('--k', type=int, default = 10, help='number of translations')
parser.add_argument('--language', type=str, default='ru', choices=['ru', 'de', 'fr'])
parser.add_argument('--gpu', type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:%s"%args.gpu if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

LANGUAGE = args.language #'ru'
en2other = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
other2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
if LANGUAGE == 'de':
    print('using german...')
    en2other = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    other2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
en2other.to(device)
other2en.to(device)

train_df = pd.read_csv(args.input)

backtr = {}
TEMPERATURE = args.temperature
print("Temperature", TEMPERATURE)
for j, row in tqdm(train_df.iterrows(), total=len(train_df)):
    try:
        target = row[args.target]
        lbl = row[args.label]
        trs = []
        if lbl!= args.skiplabel:
            for i in range(args.k):
                tr_ru = other2en.translate(en2other.translate(target, 
                                                sampling = True, 
                                                temperature = TEMPERATURE),  
                                sampling = True, 
                                temperature = TEMPERATURE)
                trs.append(tr_ru)
        backtr[j] = {
            args.target: target,
            args.label: lbl,
            'translations': [t.lower().strip() for t in trs],
            'temperature': TEMPERATURE
        }
    except Exception as e:
        print(e)
        continue
        

DIR = 'attribute_clf/backtranslations'
if not os.path.exits(DIR):
    os.makedirs(DIR)
with open(os.join(DIR, 'en_%s_%s_backtranslations.pkl' % (LANGUAGE, TEMPERATURE)), 'wb') as outfile:
    pkl.dump(backtr, outfile)