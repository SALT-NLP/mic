import pandas as pd
import numpy as np
from glob import glob
import random
import os
import argparse
from collections import Counter

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
    df['rot-bad'] = [int(row['AssignmentStatus']=='Rejected') for _,row in df.iterrows() ]
    df = df.rename(columns={
        'Input.questions': 'Q',
        'Input.A0': 'A',
        'Answer.revised_answer': 'worker_answer',
        'Answer.rot': 'rot',
        'Answer.commentbox': 'comments'
    })
    df['WorkTimeInMinutes'] = ["{:.1f}".format(time) for time in (df['WorkTimeInSeconds'] / 60).values]
    return df[['WorkerId', 'WorkTimeInMinutes', 'Q', 'A', 'rot', 'moral', 'moral-vector',
               'A_agrees', 'rot-agree', 'violation-severity', #'worker_answer', 
               'comments']]

def main():
    np.random.seed(args.seed)
    random.seed(args.seed)

    FILENAME = args.input

    out_fn = './hit/Task_SecondaryMoralEvaluation/input/%s' % FILENAME[:-4] + '_CLEANED.csv'
    if os.path.exists(out_fn):
        print('already saved file')
        return

    in_fn = os.path.join('./hit/Task_MoralEvaluation/output', FILENAME)
    print('in', in_fn)
    df = pd.read_csv(in_fn)

    print('There are', len(set(zip(df['Input.questions'].values, df['Input.A0'].values))), 'Q/A pairs')
    print('There are', len(set(df['Answer.rot'])), 'RoTs')

    c = Counter([(row['Input.questions'],row['Input.A0']) for _,row in df.iterrows()])
    fully_annotated = [key for key in c if c[key]>=3]
    fully_annotated_df = df[[(row['Input.questions'],row['Input.A0']) in fully_annotated for _,row in df.iterrows()]]

    priors = set()
    try:
        priors = set(pd.concat([pd.read_csv(fn) for fn in glob('../Task_SecondaryMoralEvaluation/output/*.csv')])['Input.rot'])
    except:
        print('no prior files')

    clean = clean_df(fully_annotated_df)
    clean = clean[[rot not in priors for rot in clean['rot'].values]].copy()
    clean.to_csv(out_fn, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='the name of the file to use as input')
    parser.add_argument('--seed', type=int, default=32)
    args = parser.parse_args()
    main()