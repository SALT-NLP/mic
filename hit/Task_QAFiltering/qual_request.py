import boto3

questions = open('qual_questions.xml', 'r').read()
answers = open('qual_answers.xml', 'r').read()


region_name = 'us-east-1'
aws_access_key_id = ''# FILL THIS IN AS A STRING
aws_secret_access_key = ''# FILL THIS IN AS A STRING

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Uncomment this line to use in production
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

mturk = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

qual_response = mturk.create_qualification_type(
                        Name='Understands QA Filtering',
                        Keywords='qualification, test, morality, psychology, nlg',
                        Description='Please answer these 6 easy yes/no questions about Q/A filtering',
                        QualificationTypeStatus='Active',
                        Test=questions,
                        AnswerKey=answers,
                        TestDurationInSeconds=7200
                        )

print(qual_response['QualificationType']['QualificationTypeId'])


