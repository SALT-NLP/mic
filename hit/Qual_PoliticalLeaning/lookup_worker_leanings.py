import boto3
import argparse
import os

region_name = 'us-east-1'
aws_access_key_id = '' # Fill this in
aws_secret_access_key = '' # Fill this in

# Uncomment this line to use in production
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

def str_base(val, base):
    res = ''
    while val > 0:
        res = str(val % base) + res
        val //= base # for getting integer division
    if res: return res
    return '0'

def average(lst):
    return sum(lst)/len(lst)

def code_to_moral_summary(q, fn="", write=True):
    code = abs(q['IntegerValue'])

    with open(fn, 'a') as outfile:

        political_mapping = {
            0: 'libertarian',
            1: 'liberal',
            2: 'moderate liberal',
            3: 'moderate conservative',
            4: 'conservative'
        }
        
        translated = str_base(code, 6)
        politics = 'libertarian'
        if len(translated)==12:
            print('politics', political_mapping[int(translated[-12])])
            politics = political_mapping[int(translated[-12])]
        else:
            print('politics', 'libertarian')
            print(len(translated))
            translated = (translated).zfill(12)
            print(translated)
        
        care = [int(translated[-1]), int(translated[-7])]
        print('care', average(care), care)
        fairness = [int(translated[-2]), int(translated[-8])]
        print('fairness', average(fairness), fairness)
        loyalty = [int(translated[-3]), int(translated[-9])]
        print('loyalty', average(loyalty), loyalty)
        authority = [int(translated[-4]), int(translated[-10])]
        print('authority', average(authority), authority)
        sanctity = [int(translated[-5]), int(translated[-11])]
        print('sanctity', average(sanctity), sanctity)
        math = int(translated[-6])
        print('math', math)

        row = [q['WorkerId'], average(care), average(fairness), average(loyalty),
               average(authority), average(sanctity), math, politics
        ]

        if write:
            outfile.write('\t'.join([str(x) for x in row])+'\n')
        
        return math

def main():
    try:
        os.remove(args.output) 
    except:
        print('Creating file from scratch.')
    with open(args.output, 'w') as outfile:
        row = ['WorkerId', 'Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity', 'Math', 'Political']
        outfile.write('\t'.join([str(x) for x in row])+'\n')
    
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    nextToken = None
    response = {}
    
    while True:
        if nextToken:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=args.qual_id,
                Status='Granted',
                NextToken=nextToken,
                MaxResults=100
            )
        else:
            response = mturk.list_workers_with_qualification_type(
                QualificationTypeId=args.qual_id,
                Status='Granted',
                MaxResults=100
            )
            
        for q in response['Qualifications']:
            print(q['WorkerId'])
            math = code_to_moral_summary(q, args.output, write=True)
            print("-----------------")
            
        if 'NextToken' not in response:
            break
        nextToken = response['NextToken']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/worker_leanings.tsv', help='the name of the tsv file to use as output')
    parser.add_argument('--qual_id', type=str, help='the name of the QualificationTypeId')
    args = parser.parse_args()
    main()