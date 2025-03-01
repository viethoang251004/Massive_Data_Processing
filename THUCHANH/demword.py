import numpy as np
import matplotlib as plt

doc ='fjoij jffjg igjifj'
words = doc.split('')
print(words)

counters = dict()
for w in words:
    if w not in counters.keys():
        counters[w] = 0
    counters[w] += 1
    
for k, v in counters.items():
    print(k, v)

def m(key,value):
    pairs = []
    for w in value.split(''):
        pairs.append((w,1))
        
pairs = m(None, doc)
print(doc)

def m(key,value):
    return [(w,1) for w in value.split('')]

pairs = m(None, doc)
print(pairs)

def g(ps):
    tmp = dict()
    for k, v in ps:
        if k not in tmp.keys():
            tmp[k] = []
        tmp[k].append(v)
    result = [(k,v) for k, v in tmp.items()]
    result.sort()
    return result

pairs = g(pairs)
print(pairs)

