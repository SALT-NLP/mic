import pandas as pd
from glob import glob
import argparse, os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def train_dev_test(df, random_state=7, dev_size=0.2, test_size=0.2, label='label'):
    df_train, df_test, y_train, y_test = train_test_split(
        df, df[label].values, test_size=test_size, 
        random_state=random_state, stratify=df[label].values
    )
    df_train, df_dev, y_train, y_dev = train_test_split(
        df_train, df_train[label].values, test_size=(dev_size/(1-test_size)) , 
        random_state=random_state, stratify=df_train[label].values
    )
    return df_train, df_dev, df_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='splits')
    parser.add_argument('--model', type=str, choices=['blenderbot', 'dialoGPT', 'gpt-neo'])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dev_size', type=int, default=0.2)
    parser.add_argument('--test_size', type=int, default=0.2)    
    args = parser.parse_args()    
    
    df = pd.concat([pd.read_csv(fn) for fn in glob(f'hit/Task_QAFiltering/output/filtered_{args.model}*.csv')+glob(f'hit/Task_QAFiltering/output/{args.model}*.csv')])
    df = df[df['AssignmentStatus']!='Rejected']
    
    columns = ['sufficient', 'moral', 'friction']
    data = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        key = (row['Input.questions'], row['Input.A0'])
        for col in columns:
            if row['Answer.%s_yes.on' % col]:
                data[key]['%s_votes' % col] += 1
            else:
                data[key]['%s_votes' % col] += 0
        data[key]['votes'] += 1
        data[key]['sentence1'] = row['Input.questions']
        data[key]['sentence2'] = row['Input.A0']
    data = pd.DataFrame().from_dict(data, orient='index').reset_index().drop(columns=['level_0', 'level_1'])

    data['sufficient'] = (data['sufficient_votes']==2).astype(int)
    data['moral'] = (data['moral_votes']>0).astype(int)
    data['friction'] = (data['friction_votes']>0).astype(int)
    data['label'] = data['moral']
    data = data.sample(frac=1, random_state=args.seed) 
    
    train, dev, test = train_dev_test(data, random_state=args.seed, dev_size=args.dev_size, test_size=args.test_size)
    print(len(train), len(dev), len(test))
    
    MORAL_DIR = os.path.join(args.output, args.model, 'moral')
    SUFFICIENT_DIR = os.path.join(args.output, args.model, 'sufficient')

    for DIR in [MORAL_DIR, SUFFICIENT_DIR]:
        if not os.path.exists(DIR):
            os.makedirs(DIR)
    
    train.to_csv( os.path.join(MORAL_DIR, 'train.csv') )
    dev.to_csv( os.path.join(MORAL_DIR, 'dev.csv') )
    test.to_csv( os.path.join(MORAL_DIR, 'test.csv') )
    
    train[train['sufficient']==1].to_csv( os.path.join(SUFFICIENT_DIR, 'train.csv') )
    dev[dev['sufficient']==1].to_csv( os.path.join(SUFFICIENT_DIR, 'dev.csv') )
    test[test['sufficient']==1].to_csv( os.path.join(SUFFICIENT_DIR, 'test.csv') )
    
if __name__=='__main__':
    main()