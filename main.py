# usage: `python3 main.py [ train_set_file ] [ test_set_file ] [ method="all" ]
import os
import sys
import numpy as np
import json
from src.cnn import *
from src.embedder import *
import random

def read_data(file_name):
    ret=[]
    with open(file_name,"r") as f:
        lines=f.readlines()
        for line in lines:
            arr=line.split(" ")
            n=len(arr)
            label={}
            for i in range(1,n):
                try:
                    tag,w=arr[i].split(":")
                    if int(w):
                        label[tag]=int(w)
                except:
                    break
            ret.append({"text":arr[i+1:],"label":label})
    return ret

if __name__=='__main__':
    if len(sys.argv)<3:
        sys.stderr.write("Too Few Arguments!!.\n")
    train_file=sys.argv[1]
    test_file=sys.argv[2]
    method=sys.argv[3] if len(sys.argv)<4 else "all"
    
    train_doc=read_data(train_file)
    random.shuffle(train_doc)
    test_doc=read_data(test_file)
    
    emb=Embedder()
    emb.train(train_doc)
    
    if method=="all" or method=="cnn":
        train_x,train_y,train_z=emb.get_embedding(train_doc,fixed_len=256)
        # with open("fuck","w") as f:
            # f.write(str(train_x[0]))
            # f.write('\n')
            # f.write(str(list(map(lambda x:emb.embedding_matrix[x],train_x[0]))))
        # exit()
        test_x,test_y,test_z=emb.get_embedding(test_doc,fixed_len=256)
        args={"fixed_len":256,"vocab_size":emb.vocab_size,"word_dim":emb.word_dim,"label_size":emb.label_size,"embedding_matrix":emb.embedding_matrix}
        # print(train_y[0:4])
        print("[ CNN ]")
        model=CnnClassifier(args,LR=0.001,epoch_size=8)
        model.train_and_test(train_x,train_z,test_x,test_z,epoch=10)
        