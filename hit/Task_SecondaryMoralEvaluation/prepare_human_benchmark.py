import pandas as pd
import os
import argparse
from ast import literal_eval

FOUNDATIONS = ['care', 'fairness', 'liberty', 'loyalty', 'authority', 'sanctity']


def main():
    out_fn = f'./hit/Task_SecondaryMoralEvaluation/input/human_benchmark.csv'
    if os.path.exists(out_fn):
        print('already saved file')
        return
    
    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn))
    
    moral_eval_df = pd.read_csv(args.input)
    for j, which in enumerate(FOUNDATIONS):
        moral_eval_df[which] = [ literal_eval(mv)[j] for mv in moral_eval_df['moral-vector'].values ]
        
    sample = moral_eval_df[moral_eval_df['split']=='test'].sample(n=args.nsamples, random_state=args.seed).copy()
    sample.to_csv(out_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='the name of the file to use as input', default='./data/mic/MIC_oct23.csv')
    parser.add_argument('--nsamples', type=int, help='the number of datapoints to use', default=300)
    parser.add_argument('--seed', type=int, default=32)
    args = parser.parse_args()
    main()
