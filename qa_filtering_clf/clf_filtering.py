import pandas as pd
import numpy as np

from collections import Counter
from glob import glob
import pandas as pd
import numpy as np
from nltk import ngrams
import os, argparse, nltk, re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='dialoGPT')
    args = parser.parse_args()

    df = pd.read_csv(f'data/prepared/{model}_generations_agg/filtered.csv')
    df['moral_proba'] = np.array([float(x.strip()) for x in open(f"qa_filtering_clf/{model}_filtered_moral_predictions.txt", 'r').readlines()])
	df['sufficient_proba'] = np.array([float(x.strip()) for x in open(f"qa_filtering_clf/{model}_filtered_sufficient_predictions.txt", 'r').readlines()])

	df[(df['sufficient_proba']>0.5) & (df['moral_proba']>0.5)].sample(frac=1, random_state=7).to_csv(f'data/prepared/{model}_generations_agg/{model}_filtered_clf_sufficient_moral.csv')
    
if __name__=='__main__':
    main()