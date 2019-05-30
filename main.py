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
    method=sys.argv[3] if len(sys.argv)>=4 else "all"
    emb_source=sys.argv[4] if len(sys.argv)>=5 else "train"
    print(emb_source)
    
    train_doc=read_data(train_file)
    random.shuffle(train_doc)
    test_doc=read_data(test_file)
    
    if emb_source=="train":
        emb=Embedder()
        emb.train(train_doc+test_doc,model_file="data/sgns.sogounews.bigram-char")
        emb.dump("emb.json")
    elif emb_source=="load":
        emb=Embedder()
        emb.load("emb.json")
    else:
        emb=Embedder()
        emb.train(train_doc)
    
    if method in ["all","mlp","cnn","textcnn"]:
        fixed_length=512
        train_x,train_y,train_z=emb.get_embedding(train_doc,fixed_len=fixed_length)
        test_x,test_y,test_z=emb.get_embedding(test_doc,fixed_len=fixed_length)
        args={"fixed_len":fixed_length,"vocab_size":emb.vocab_size,"word_dim":emb.word_dim,"label_size":emb.label_size,"embedding_matrix":emb.embedding_matrix}
    if method=="all" or method=="mlp":
        print("{ **MLP** }")
        model=Classifier(args,LR=0.0005,batch_size=8,network="mlp")
        model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=30)
    if method=="all" or method=="cnn":
        print("{ **CNN** }")
        model=Classifier(args,LR=0.0005,batch_size=8,network="cnn")
        model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=30)
    if method=="all" or method=="textcnn":
        print("{ **TextCNN** }")
        model=Classifier(args,LR=0.0001,batch_size=4,network="textcnn")
        model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=100)
    
    if method in ["all","rnn","gru"]:
        train_x,train_y,train_z=emb.get_embedding(train_doc)
        test_x,test_y,test_z=emb.get_embedding(test_doc)
        args={"vocab_size":emb.vocab_size,"word_dim":emb.word_dim,"label_size":emb.label_size,"embedding_matrix":emb.embedding_matrix}
    if method=="all" or method=="rnn":
        print("{ **RNN** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="rnn")
        model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=100)
    if method=="all" or method=="gru":
        print("{ **GRU** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="gru")
        model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=100)