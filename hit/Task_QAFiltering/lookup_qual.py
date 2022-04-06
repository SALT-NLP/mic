def str_base(val, base):
    res = ''
    while val > 0:
        res = str(val % base) + res
        val //= base # for getting integer division
    if res: return res
    return '0'

def average(lst):
    return sum(lst)/len(lst)

def code_to_moral_summary(q, fn=""):
    code = q['IntegerValue']

    with open(fn, 'a') as outfile:

        political_mapping = {
            0: 'libertarian',
            1: 'liberal',
            2: 'moderate liberal',
            3: 'moderate conservative',
            4: 'conservative'
        }
        
        translated = str_base(code, 6)
        if len(translated)==12:
            print('politics', political_mapping[int(translated[-12])])
        else:
            print('politics', 'libertarian')
            print(len(translated))
            translated = (translated).zfill(12)
            print(translated)
        
        harm = [int(translated[-1]), int(translated[-7])]
        print('harm', average(harm), harm)
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

        row = [average(harm), average(fairness), average(loyalty),
               average(authority), average(sanctity), math, translated[-12]
        ]

        outfile.write('\t'.join([str(x) for x in row])+'\n')

import boto3

region_name = 'us-east-1'
aws_access_key_id = 'AKIA5KK5LWXAPDWDI77N'
aws_secret_access_key = '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'

# Uncomment this line to use in production
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

mturk = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

response = mturk.list_workers_with_qualification_type(
    QualificationTypeId='32IU3MUE9F1JVFYNOIG3JU6V48L6EI',
    Status='Granted',
    #NextToken='p2:7fJ5BPRHR65o/542b/oitzJJc8JRg1qw/HFRB0LX5Pc8QL+7e5hEQeLHQIYDnSo=', # feel free to remove this
    MaxResults=100
)

#print(response)
for q in response['Qualifications']:
    print(q['WorkerId'])
    code_to_moral_summary(q, 'worker_leanings.tsv')
    print("-----------------")
print('next', response['NextToken'])

