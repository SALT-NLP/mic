import pandas as pd
from glob import glob
import argparse, os, re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', type=str)
    parser.add_argument('--input_glob', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--nsamples', type=int, default=25)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.mic)
    test = df[df['split']=='test'].copy()

    N_SAMPLES = args.nsamples
    samples = []
    process = lambda txt : re.sub('[\s]*\[[\w/]+\][\s]*', '', re.sub('[\s]*<[\w/]+>[\s]*', '', 
                      re.sub('^([\s]*<[\w/]+>[\s]*)+', '', txt).split('<eos>')[0].split('</s>')[0]).strip())
    for fn in sorted(glob(args.input_glob)):
        tag = '_'.join(fn.split('/')[-1].split('_')[2:-1])
        model='t5'
        if 'GROUND_TRUTH' in fn:
            model='human'
        elif 'SBERT' in fn:
            model='SBERT_retrieval'
        elif 'retrieval' in fn and 'random' in fn:
            model = 'random_retrieval'
        elif 'gpt' in fn:
            model='gpt'
        elif 'bart' in fn:
            model='bart'
        gens = pd.read_csv(fn)
        gens['Q'] = test['Q'].values
        gens['A'] = test['A'].values
        gens['model']=model
        gens['tag'] = tag
        gens = gens[['Q', 'A', 'rot_generated', 'model', 'tag']].copy()
        gens['rot_generated'] = [process(txt) for txt in gens['rot_generated'].values]
        samples.append(gens.sample(n=N_SAMPLES, random_state=args.seed))

        out = pd.concat(samples).sample(frac=1, random_state=args.seed)
        out.to_csv(args.output, index=False)

if __name__=='__main__':
    main()
