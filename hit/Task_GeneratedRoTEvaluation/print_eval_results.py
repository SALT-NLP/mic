import pandas as pd
import numpy as np
import simpledorff
import pingouin as pg
import argparse

def to_scale(row, col_name):
    for val in range(1,6):
        if row[f'{col_name}_{val}.on']:
            return val
    return -1

def agrees(row):
    if row['Answer.A_agrees.on']:
        return 2#'agrees'
    elif row['Answer.A_disagrees.on']:
        return 0#'disagrees'
    else:
        return 1#'neutral'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['A_agrees'] = [agrees(row) for _,row in df.iterrows()]
    df['relevance'] = [to_scale(row, col_name='Answer.relevance') for _,row in df.iterrows()]
    df['rot_consensus'] = [to_scale(row, col_name='Answer.rot_consensus') for _,row in df.iterrows()]
    df['fluency'] = [to_scale(row, col_name='Answer.fluency') for _,row in df.iterrows()]
    df['wellformed'] = df['Answer.wellformed_yes.on'].values.astype(int)

    for input_name in ['Input.Q', 'Input.A',
           'Input.rot_generated', 'Input.model', 'Input.tag']:
        df[input_name.replace("Input.", "")] = df[input_name]

    df = df.sort_values(by='rot_generated')
    df['judge'] = np.tile([1,2,3], int(len(df)/3))
    clean = df[['WorkerId', 'HITId', 'judge', 'Q', 'A',
                'rot_generated', 'model', 'tag', 
                'A_agrees', 'relevance', 'rot_consensus', 'fluency', 'wellformed']]

    print('---Inter annotator agreement---')
    CONSTRUCTS = ['A_agrees', 'relevance', 'rot_consensus', 'fluency', 'wellformed']
    CONSTRUCT_NAMES = ['Answer Alignment', 'Relevance', 'Global Consensus', 'Fluence', 'Well-Formed']
    for construct, construct_name in zip(CONSTRUCTS, CONSTRUCT_NAMES):
        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(clean,
                                                         experiment_col='HITId',
                                                         annotator_col='WorkerId',
                                                         class_col=construct)
        icc = pg.intraclass_corr(data=clean, targets='HITId', raters='judge',
                             ratings=construct, nan_policy='omit').round(3).set_index('Type').loc['ICC1k']['ICC']
        print(f"{construct_name} & {alpha:.2f} & {icc:.2f}\\\\")

    print()
    print('---Human Evaluation---')
    print("Model & Decoding & Well-Formed & Fluent & Relevant \\")
    results = clean.groupby(['model', 'tag']).mean()
    for model in ['bart', 't5', 'gpt']:
        for tag in ['beams0_p0_k0', 'beams3_p0_k0', 'beams0_p0.9_k0']:
            row = results.loc[model].loc[tag]
            print(f"{model} & {tag} & {row['wellformed']:.2f} & {row['fluency']:.2f} & {row['relevance']:.2f} \\\\")
            
    row = results.loc['SBERT_retrieval'].iloc[0]
    print(f"SBERT &  & {row['wellformed']:.2f} & {row['fluency']:.2f} & {row['relevance']:.2f} \\\\")

    row = results.loc['random_retrieval'].iloc[0]
    print(f"Random RoT &  & {row['wellformed']:.2f} & {row['fluency']:.2f} & {row['relevance']:.2f} \\\\")

    row = results.loc['human'].iloc[0]
    print(f"Human &  & {row['wellformed']:.2f} & {row['fluency']:.2f} & {row['relevance']:.2f} \\\\")


if __name__=='__main__':
    main()