# usage: `python3 main.py [ train_set_file ] [ test_set_file ] [ method="all" ]
import os
import sys
import numpy as np
import json
from src.models import *
from src.embedder import *
from src.svm import *
import random

def _p(buf):
    print(buf)
    with open("result.log.txt","a") as f:
        f.write(str(buf)+'\n')

def read_data(file_name):
    ret=[]
    with open(file_name,"r") as f:
        lines=f.readlines()
        for line in lines:
            _,tags,content=line.split("\t")
            tags=tags.split(" ")
            content=content.split(" ")
            label={}
            for buf in tags[1:]:
                tag,w=buf.split(":")
                if int(w):
                    label[tag]=int(w)
            ret.append({"text":content,"label":label})
    return ret

if __name__=='__main__':
    if len(sys.argv)<3:
        sys.stderr.write("Too Few Arguments!!.\n")
    train_file=sys.argv[1]
    test_file=sys.argv[2]
    method=sys.argv[3] if len(sys.argv)>=4 else "all"
    mode=sys.argv[4] if len(sys.argv)>=5 else "test"
    print(mode)
    
    train_doc=read_data(train_file)
    random.shuffle(train_doc)
    test_doc=read_data(test_file)
    
    # SVM
    if method=="svm":
        _p("{ **SVM** }")
        emb_tfidf=Embedder(method="tf-idf")
        emb_tfidf.train(train_doc)
        train_x,_,train_y=emb_tfidf.get_embedding(train_doc)
        test_x,test_y,_=emb_tfidf.get_embedding(test_doc)
        svm=Svm()
        svm.train_and_test(train_x,train_y,test_x,test_y)
        exit()
    # SVM
    
    if mode=="train":
        emb=Embedder()
        emb.train(train_doc+test_doc,model_file="data/sgns.sogounews.bigram-char")
        emb.dump("emb.json")
    elif mode in ["load","draw"]:
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
        _p("{ **MLP** }")
        model=Classifier(args,LR=0.0005,batch_size=8,network="mlp")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=5)
    if method=="all" or method=="cnn":
        _p("{ **CNN** }")
        model=Classifier(args,LR=0.0005,batch_size=8,network="cnn")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=30)
    if method=="all" or method=="textcnn":
        _p("{ **TextCNN** }")
        model=Classifier(args,LR=0.0002,batch_size=4,network="textcnn")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=50)
    
    if method in ["all","rnn","gru","lstm","rcnn","clstm"]:
        train_x,train_y,train_z=emb.get_embedding(train_doc)
        test_x,test_y,test_z=emb.get_embedding(test_doc)
        args={"vocab_size":emb.vocab_size,"word_dim":emb.word_dim,"label_size":emb.label_size,"embedding_matrix":emb.embedding_matrix}
    if method=="all" or method=="rnn":
        _p("{ **RNN** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="rnn")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=60)
    if method=="gru": # ignored
        _p("{ **GRU** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="gru")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=100)
    if method=="lstm": # ignored
        _p("{ **LSTM** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="lstm")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=100)
    if method=="all" or method=="rcnn":
        _p("{ **RCNN** }")
        model=Classifier(args,LR=0.0001,batch_size=1,network="rcnn")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=15)
    if method=="all" or method=="clstm":
        _p("{ **C-LSTM** }")
        model=Classifier(args,LR=0.0002,batch_size=1,network="clstm")
        if mode=="draw":
            model.draw(train_x)
        else:
            model.train_and_test(train_x,train_y,test_x,test_y,test_z,epoch=30)