# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import multiprocessing
import re

from collections import defaultdict
import argparse
import os

def get_moral_dict(path='resources/Enhanced_Morality_Lexicon_V1.1.txt'):
    morals = defaultdict(set)
    with open(path, 'r') as infile:
        for line in infile.readlines():
            split=line.split('|')
            token=split[0][8:]
            moral_foundation = split[4][9:]
            morals[moral_foundation].add(token)
    return dict(morals)

def setdict_to_regexes(dic):
    regexes = {}
    for key in dic.keys():
        regexes[key] = r'\b' + r'\b|\b'.join([element.lower() for element in dic[key]]) + r'\b'
    return regexes

def worker(name, df, regexes, args):
    which=args.column
    print('starting', name)
    arr = np.array([0 for i in range(len(df))])
    for i, txt in tqdm(enumerate(df[which].fillna('').values), total=len(df)):
        arr[i] = len(re.findall(regexes[name], txt, flags=re.IGNORECASE))

    if len(args.indir):
        outdir = args.indir+"_agg"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        np.save(os.path.join(outdir, "%s_%s.npy" % (which, name)), arr)
    else:
        np.save('data/raw/Reddit-QA-Corpus/%s_%s.npy' % (which,name), arr)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="")
    parser.add_argument("--column", type=str, default='questions')
    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")
    args = parser.parse_args()

    questions = open('data/raw/Reddit-QA-Corpus/Questions_R.txt', 'r').readlines()
    answers = open('data/raw/Reddit-QA-Corpus/Answers_R.txt', 'r').readlines()
    df=pd.DataFrame().from_dict({'questions': questions, 'answers': answers}, orient='columns')

    if len(args.indir):
        print("using", args.indir)
        df = pd.concat([pd.read_csv(fn) for fn in sorted(glob( os.path.join(args.indir, '*.csv')  ))])

    regexes = setdict_to_regexes(get_moral_dict())
    jobs = []

    if args.multiprocessing:
        for regex in regexes:
            p = multiprocessing.Process(target=worker, args=(regex, df, regexes, args))
            jobs.append(p)
            p.start()
    else:
        for regex in regexes:
            worker(regex, df, regexes, args)
