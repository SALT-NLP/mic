from glob import glob
import pandas as pd
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['blenderbot', 'dialoGPT', 'gpt-neo'])
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    args = parser.parse_args()

    model = args.model

    start_idx = args.start_idx
    end_idx = args.end_idx
    
    out_fn = 'hit/Task_QAFiltering/input/filtered_%s_%s_%s.csv' % (model, start_idx, end_idx)
    
    if os.path.exists(out_fn):
        print('already prepared', out_fn)
        return

    prior_pairs = set()
    try:
        prior = pd.concat([pd.read_csv(fn) for fn in glob('input/*%s*.csv' % model)])
        prior_pairs = set([(row['questions'], row['%s_A0' % model]) for _,row in prior.iterrows()])
    except:
        pass
    
    out = pd.read_csv('data/prepared/%s_generations_agg/filtered.csv' % model)
    out = out.sample(frac=1, random_state=38).iloc[start_idx:end_idx][['questions', 'answers', '%s_A0' % model]]
    out = out.rename(columns={'%s_A0' % model: 'A0'})

    out = out[[(row['questions'], row['A0']) not in prior_pairs for _,row in out.iterrows()]].copy()
    len(out)
    out.to_csv(out_fn, index=False)


if __name__=='__main__':
    main()
