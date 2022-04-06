# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import glob
import os
import numpy as np
import pandas as pd
questions = open('data/raw/Reddit-QA-Corpus/Questions_R.txt', 'r').readlines()
answers = open('data/raw/Reddit-QA-Corpus/Answers_R.txt', 'r').readlines()
df=pd.DataFrame().from_dict({'questions': questions, 'answers': answers}, orient='columns').iloc[:10]

df['reddit'] = df['questions'].str.contains(r'reddit', case=False, regex=True)
df['is_question'] = df['questions'].str.contains('?', regex=False)

for fn in glob.glob('data/raw/Reddit-QA-Corpus/*.npy'):
    basename = os.path.basename(fn)[:-4]
    df[basename] = np.load(fn)

q = df[['questions_GeneralVice',
       'questions_GeneralVirtue', 'questions_FairnessVirtue',
       'questions_AuthorityVirtue', 'questions_CareVirtue',
       'questions_PurityVice', 'questions_IngroupVice',
       'questions_FairnessVice', 'questions_CareVice',
       'questions_PurityVirtue', 'questions_IngroupVirtue',
       'questions_AuthorityVice']].sum(axis=1)

a = df[['answers_FairnessVirtue', 'answers_FairnessVice', 'answers_GeneralVirtue',
        'answers_PurityVirtue', 'answers_CareVice',
       'answers_CareVirtue', 'questions_CareVice', 'answers_AuthorityVice',
       'answers_AuthorityVirtue', 'answers_IngroupVice', 'answers_IngroupVirtue', 'answers_PurityVice',
       'answers_GeneralVice']].sum(axis=1)

if not os.path.exists('data/raw/Reddit-QA-Corpus/filtered/'):
    os.makedirs('data/raw/Reddit-QA-Corpus/filtered/')
consider = df[(q>1) & (a>1) & (~df['reddit']) & (df['is_question'])].copy()
for i in range(int(len(consider)/100)+1):
    sub = consider.iloc[i*100:(i+1)*100].to_csv('data/raw/Reddit-QA-Corpus/filtered/%s_%s.csv' % (i*100, (i+1)*100), index=False)
