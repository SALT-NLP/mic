from collections import Counter
import pandas as pd
import numpy as np
from glob import glob
import random
import argparse, os
np.random.seed(32)
random.seed(32)

def moral_foundations(row):
    mfs = []
    for which in ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']:
        if row['Answer.%s.on' % which]:
            mfs.append(which)
    return '|'.join(mfs)

def rot_consensus(row):
    for grade in range(1,6):
        if row['Answer.rot_consensus_%s.on' % grade]:
            return grade
        
def violation_severity(row):
    for grade in range(1,6):
        if row['Answer.violation_severity_%s.on' % grade]:
            return grade
        
def clean_df(df):
    idx={val:i for i, val in enumerate(sorted(list(set([row['Input.rot'] for _,row in df.iterrows()]))))}
    df['rot_id'] = [idx[row['Input.rot']] for _,row in df.iterrows()]

    for which in ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']:
        df[f'{which}_AUTHOR'] = [int(which in row['Input.moral']) if type(row['Input.moral'])==str else 0 for _, row in df.iterrows()]

    for which in ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']:
        df[which] = (df['Answer.%s.on' % which]==True).astype(int)
    
    df['moral'] = [moral_foundations(row) for _, row in df.iterrows()]
    df['rot_consensus'] = [rot_consensus(row) for _, row in df.iterrows()]
    df['violation_severity'] = [violation_severity(row) for _,row in df.iterrows()]
    df = df.rename(columns={
       'Input.rot': 'rot'
    })
    df['violation_severity_AUTHOR'] = df['Input.violation-severity']
    df['rot_consensus_AUTHOR'] = df['Input.rot-agree']
    df['WorkTimeInMinutes'] = ["{:.1f}".format(time) for time in (df['WorkTimeInSeconds'] / 60).values]
    return df[['HITId', 'WorkerId', 'WorkTimeInMinutes', 'rot_id', 'rot', 'moral', 
               'rot_consensus', 'violation_severity', 'care', 'fairness', 
               'rot_consensus_AUTHOR', 'violation_severity_AUTHOR',
               'liberty', 'loyalty', 'authority', 'sanctity', 'care_AUTHOR', 
               'fairness_AUTHOR', 'liberty_AUTHOR', 'loyalty_AUTHOR', 
               'authority_AUTHOR', 'sanctity_AUTHOR']]

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    df = pd.concat([
        pd.read_csv(fn) for fn in glob('./hit/Task_SecondaryMoralEvaluation/output/*.csv')
    ])
    df = df[df['AssignmentStatus']!='Rejected']

    clean = clean_df(df)
    
    clean.to_csv(args.output)
    
if __name__=='__main__':
    main()