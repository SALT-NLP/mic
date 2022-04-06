import pickle as pkl
import numpy as np
import pandas as pd
from glob import glob

cumulative = {}
for fn in glob('attribute_clf/backtranslations/*.pkl'):
    backtr = pkl.load(open(fn, 'rb'))
    for key in backtr:
        targ = backtr[key]['rot']
        if not targ in cumulative:
            cumulative[targ] = {
                'translations': set(backtr[key]['translations'])
            }
        else:
            cumulative[targ]['translations'].update(set(backtr[key]['translations']))
    print(fn)

cumulative = {
    i: {
        'rot': key,
        'translations': list(cumulative[key]['translations'])
    }
    
    for i, key in enumerate(cumulative)
}

with open('attribute_clf/backtranslations/mixed_en_de_ru_backtranslations.pkl', 'wb') as outfile:
    pkl.dump(cumulative, outfile)