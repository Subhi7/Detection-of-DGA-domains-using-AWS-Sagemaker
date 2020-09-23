import json
import logging
import boto3
import time
import tldextract
from collections import Counter
from itertools import groupby
import math

from pprint import pprint

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def predict_one_dga_value(sm_client, features, endpoint_name):
    # print('Using model endpoint {} to predict dga for this feature vector: {}'.format(endpoint_name, features))
    is_dga = False
    body = features + '\n'
    start_time = time.time()

    response = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=body)
    predicted_value = json.loads(response['Body'].read())
    duration = time.time() - start_time
    if predicted_value > 0.5:
        is_dga = True
    return is_dga


LOOKUP_TABLE = {'a': 5, 'b': 9, 'c': 17, 'd': 30, 'e': 22, 'f': 2, 'g': 35, 'h': 19, 
            'i': 12, 'j': 28, 'k': 20, 'l': 24, 'm': 10, 'n': 13, 'o': 7, 
            'p': 26, 'q': 4, 'r': 37, 's': 11, 't': 15, 
            'u': 16, 'v': 25, 'w': 6, 'x': 8, 'y': 1, 'z': 3, 
            '0': 36, '1': 23, '2': 31, '3': 33, '4': 27, '5': 29, 
            '6': 38, '7': 32, '8': 14, '9': 21, '-': 39, '.': 18, '_':34}

def calc_entropy(s):
    p, lns = Counter(str(s)), float(len(str(s)))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def capital(z):
    num_digits = 0
    for char in z:
        if char.isupper():
            num_digits += 1
    return num_digits
    
def calc_digits(z):
    num_digit = 0
    digitslist = list('0123456789')
    for char in z:
        if char in digitslist:
            num_digit += 1
    return num_digit

def consecutive_consonants(string):
    is_vowel = lambda char: char in "aAeEiIoOuU"
    best = 0
    listnames = ["".join(g) for v, g in groupby(string, key=is_vowel) if not v]
    for index in range(len(listnames)):
        if len(listnames[index]) > best:
            best = len(listnames[index])
    return best
    
def calc_vowels(y):
    num_vowel = 0
    vowels = list('aeiou')
    for char in y:
        if char in vowels:
            num_vowel += 1
    return num_vowel

def unique_char(x):
    return len(''.join(set(x)))

def encode_fqdn(fqdn='www.google.com'):
    global LOOKUP_TABLE

    ds = tldextract.extract(fqdn)
    domain1 = ds.domain
    capit = capital(domain1)
    domain = domain1.lower()
    length = len(domain)
    dig = calc_digits(domain)
    cons = consecutive_consonants(domain)
    nv = calc_vowels(domain)
    nv_divided = nv/length
    entrop1 = calc_entropy(domain)
    unique1 = unique_char(domain)
    rvalue = list()
    for c in domain:
        rvalue.append(str(LOOKUP_TABLE[c]))
    for _ in range(len(rvalue), 64):
        rvalue.insert(0,'0')
    rvalue.append(str(length))
    rvalue.append(str(capit))
    rvalue.append(str(dig))
    rvalue.append(str(cons))
    rvalue.append(str(nv_divided))
    rvalue.append(str(entrop1))
    rvalue.append(str(unique1))
    return ','.join(rvalue)

def lambda_handler(event, context):
    runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
    if 'queryStringParameters' not in event:
        return {
            'statusCode': 227,
            'body': json.dumps('MISSING queryStringParameters')
        }
    else:
        query_parms = event['queryStringParameters']
        logger.info(msg=query_parms)
        features = encode_fqdn(fqdn=query_parms['fqdn'])
        
        p = predict_one_dga_value(sm_client=runtime_sm_client, features=features, endpoint_name='DEMO-XGBoostEndpoint-2020-05-30-00-55-08')
        query_parms['dga'] = p
        logger.info(msg=query_parms)
        return {
            'statusCode': 200,
            'body': json.dumps(query_parms)
        }