from collections import Counter
from glob import glob
import pandas as pd
import numpy as np
from nltk import ngrams
import os, argparse, nltk, re

def is_empty_response(response):
    return (len(re.findall(r"(I am|I'm) not sure", response))>0) or \
        (len(re.findall(r"I think (it was a joke|you mean)", response))>0)

def count_most_common_ngram(sentence, n=3):
    try:
        return Counter(list(ngrams(nltk.word_tokenize(sentence), n))).most_common()[0][1]
    except:
        return 0
    
def filter_pipeline(df, col):
    df_ = df[[(not is_empty_response(x)) for x in df[col].values]].copy()
    df_ = df_[[count_most_common_ngram(txt, 5)<=1 for txt in df_[col].values]].copy()
    return df_

def filter_responses(base, col):
    df = pd.concat([pd.read_csv(fn) for fn in sorted(glob('%s/*' % base))])
    for fn in glob('%s_agg/*.npy' % base):
        basename = os.path.basename(fn)[:-4]
        df[basename] = np.load(fn)
    df.to_csv('%s_agg/all.csv' % base)
    columns = ['FairnessVirtue','AuthorityVirtue', 'CareVirtue','PurityVice', 'IngroupVice',
           'FairnessVice', 'CareVice','PurityVirtue', 'IngroupVirtue','AuthorityVice']
    r = df[['%s_%s' % (col, c) for c in columns]].sum(axis=1)
    filter_pipeline(df[r>0], col).to_csv('%s_agg/filtered.csv' % base) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default='data/dialoGPT_generations_ORIGINAL_2')
    parser.add_argument("--column", type=str, default='dialoGPT_A0')
    args = parser.parse_args()
    
    filter_responses(args.indir, args.column)
    
if __name__=='__main__':
    main()