import sys
import os
import numpy as np
import json
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

def accuracy(y1,y2):
    n=min(len(y1),len(y2))
    acc=0.0
    for i in range(n):
        if y1[i]==y2[i]:
            acc+=1/n
    return acc

def macro_f1(y1,y2):
    return f1_score(y1,y2,average="macro")

'''
def f_score(y1,y2):
    max_lab=0
    n=min(len(y1),len(y2))
    for i in range(n):
        max_lab=max(max_lab,max(y1[i],y2[i]))
    max_lab+=1
    cnt1=[0 for i in range(max_lab)]
    cnt2=[0 for i in range(max_lab)]
    correct=[0 for i in range(max_lab)]
    # print([cnt1,cnt2,correct])
    for i in range(n):
        cnt1[y1[i]]+=1
        cnt2[y2[i]]+=1
        if y1[i]==y2[i]:
            correct[y1[i]]+=1
    ret=0.0
    for i in range(max_lab):
        p=correct[i]/cnt1[i] if cnt1[i] else 1
        r=correct[i]/cnt2[i] if cnt2[i] else 1
        f=2*p*r/(p+r) if (p+r) else 0
        ret+=f/max_lab
    return ret
'''

def corr(y1,y2):
    n=min(len(y1),len(y2))
    ret=0.0
    for i in range(n):
        ret+=pearsonr(y1[i],y2[i])[0]/n
    return ret