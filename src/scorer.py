import sys
import os
import numpy as np
import json


def accuracy(y1,y2):
    n=min(len(y1),len(y2))
    acc=0.0
    for i in range(n):
        if y1[i]==y2[i]:
            acc+=1/n
    return acc

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
            correct[i]+=1
    ret=0.0
    for i in range(max_lab):
        p=correct[i]/cnt1[i]
        r=correct[i]/cnt2[i]
        f=p*r/(p+r)
        ret+=f/max_lab
    return ret