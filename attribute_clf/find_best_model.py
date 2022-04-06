import json, argparse, os, re
from glob import glob
import numpy as np

def macro(j, which='f1'):
    regex = f"{which}_[0-9]+_1"
    
    vals = []
    for key in j:
        if re.match(regex, key):
            vals.append(j[key])
    return np.mean(np.array(vals))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='directory to scan')
    parser.add_argument('--metric', type=str, default='f1', help='metric to use when sorting')
    parser.add_argument('--negate', action='store_true', help='whether to sort using the negative version of the metric (minimize)')
    parser.add_argument('--macro', action='store_true')
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()
    
    scale = 1.0
    if args.negate:
        scale = -1.0
   
    optimal_fn = ""
    optimal_metric = -1e9
    optimal_json = {}
    for fn in glob(os.path.join(args.dir, f'{args.model}*.json')):
        j = json.load(open(fn, 'r'))
        val = 0
        if args.macro:
            val = macro(j, args.metric)*scale
        else:
            val = j[args.metric]*scale
        print(',', fn, val)
        if val>=optimal_metric:
            optimal_fn = fn
            optimal_json = j
            optimal_metric = val
            
    print(optimal_metric, optimal_fn)