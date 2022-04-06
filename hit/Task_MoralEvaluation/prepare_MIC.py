from collections import Counter
import pandas as pd
import numpy as np
from glob import glob
import random
import argparse, os
np.random.seed(32)
random.seed(32)

def read_annotations(fn):
    df = pd.read_csv(fn)
    chatbot = 'unknown'
    if 'blenderbot' in fn:
        chatbot = 'blenderbot'
    elif 'dialoGPT' in fn:
        chatbot = 'dialoGPT'
    elif 'gpt-neo' in fn:
        chatbot = 'gpt-neo'
    df['chatbot'] = chatbot
    return df

def moral_foundations(row):
    mfs = []
    for which in ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']:
        if row['Answer.%s.on' % which]:
            mfs.append(which)
    if not len(mfs):
        return None
    return '|'.join(mfs)

def moral_vector(row):
    mfs = []
    for which in ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']:
        if row['Answer.%s.on' % which]:
            mfs.append(1.0)
        else:
            mfs.append(0.0)
    return mfs

def rot_consensus(row):
    for grade in range(1,6):
        if row['Answer.rot_consensus_%s.on' % grade]:
            return grade
    return -1
        
def violation_severity(row):
    for grade in range(1,6):
        if row['Answer.violation_severity_%s.on' % grade]:
            return grade
    return -1
        
def agrees(row):
    if row['Answer.A_agrees.on']:
        return 2#'agrees'
    elif row['Answer.A_disagrees.on']:
        return 0#'disagrees'
    else:
        return 1#'neutral'
        
def clean_df(df):
    df['moral'] = [moral_foundations(row) for _, row in df.iterrows()]
    df['moral-vector'] = [moral_vector(row) for _, row in df.iterrows()]
    df['rot-agree'] = [rot_consensus(row) for _, row in df.iterrows()]
    df['violation-severity'] = [violation_severity(row) for _,row in df.iterrows()]
    df['A_agrees'] = [agrees(row) for _, row in df.iterrows()]
    
    collapsed_names = []
    for which in ['violation-severity', 'rot-agree']:
        # cut off bottom two values {1,2} for the ordinal scales and shift labels from {3,4,5} down to {1,2,3}
        collapsed_names.append(f'{which}-collapsed')
        df[f'{which}-collapsed'] = (df[which]-2).clip(1,3)
    df = df.rename(columns={
        'Input.questions': 'Q',
        'Input.A0': 'A',
        'Answer.revised_answer': 'worker_answer',
        'Answer.rot': 'rot',
        'Answer.commentbox': 'comments'
    })
    df['WorkTimeInMinutes'] = ["{:.1f}".format(time) for time in (df['WorkTimeInSeconds'] / 60).values]
    
    # remove newlines / carriage returns from ROT
    df['rot'] = df['rot'].replace(r'\r',' ', regex=True)
    df['rot'] = df['rot'].replace(r'\n',' ', regex=True)
    return df[['WorkerId', 'WorkTimeInMinutes', 'chatbot', 'Q', 'A', 'rot', 'moral', 'moral-vector',
               'A_agrees', 'rot-agree', 'violation-severity', #'worker_answer', 
               'comments'] + collapsed_names]

def add_split_col_and_shuffle(df, no_bleeding, train_frac=0.8, dev_frac=0.1):
    train, dev, test = train_dev_test(df, no_bleeding, train_frac, dev_frac)
    train['split'] = 'train'
    dev['split'] = 'dev'
    test['split'] = 'test'
    return pd.concat([train,dev,test]).sample(frac=1, random_state=0)
    

def train_dev_test(df, no_bleeding, train_frac=0.8, dev_frac=0.1):
    assert 1-train_frac-dev_frac > 0
    train, test = split(df, frac=train_frac, no_bleeding=no_bleeding)
    dev, test = split(test, frac=(dev_frac/(1-train_frac)), no_bleeding=no_bleeding)
    return train, dev, test
    
def split(df, frac=0.8, no_bleeding=''):
    if not (len(no_bleeding) and no_bleeding in df):
        print('fix no_bleeding', no_bleeding)
        
    pool = set(df[no_bleeding].values)
    train_pool = set(random.sample(sorted(list(pool)), k=int(len(pool)*frac)))
    test_pool = pool.difference(train_pool)
   
    filt = lambda s : df[[val in s for val in df[no_bleeding].values]].copy()    
    return filt(train_pool), filt(test_pool)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    df = pd.concat([
        read_annotations(fn) for fn in glob('./hit/Task_MoralEvaluation/output/*.csv')
    ])
    df = df[df['AssignmentStatus']!='Rejected']

    print('There are', len(set(zip(df['Input.questions'].values, df['Input.A0'].values))), 'Q/A pairs')
    print('There are', len(set(df['Answer.rot'])), 'RoTs')

    c = Counter([(row['Input.questions'],row['Input.A0']) for _,row in df.iterrows()])
    fully_annotated = [key for key in c if c[key]>=3]
    fully_annotated_df = df[[(row['Input.questions'],row['Input.A0']) in fully_annotated for _,row in df.iterrows()]].copy()

    clean = clean_df(fully_annotated_df)

    clean['QA'] = [f"Q: {row['Q'].strip()} A: {row['A'].strip()}" for _, row in clean.iterrows() ]
    
    s = add_split_col_and_shuffle(clean, no_bleeding='QA')
    
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    s.to_csv(args.output)
    
if __name__=='__main__':
    main()
