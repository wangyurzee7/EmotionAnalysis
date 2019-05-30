#coding=utf-8
import sys
import os
import gensim
import numpy as np
from gensim.models.word2vec import Word2Vec
# from gensim.models import *
import json

class Embedder:
    def __init__(self,method="word2vec"):
        self.method=method
        
        self.trained=False
        # self.model=None
        # self.word_dim=None
        # self.labels=None
        # self.label2id=None
        # self.label_size=None
        # self.vocabs=None
        # self.vocab2id=None
        # self.vocab_size=None
        # self.embedding_matrix=None
    
    def dump(self,file_name):
        tmp={
            "word_dim":self.word_dim,
            "labels":self.labels,
            "label2id":self.label2id,
            "label_size":self.label_size,
            "vocabs":self.vocabs,
            "vocab2id":self.vocab2id,
            "vocab_size":self.vocab_size,
            "embedding_matrix":self.embedding_matrix.tolist()
        }
        with open(file_name,"w") as f:
            json.dump(tmp,f)
    
    def load(self,file_name):
        with open(file_name,"w") as f:
            tmp=json.load(f)
        self.word_dim=tmp["word_dim"]
        self.labels=tmp["labels"]
        self.label2id=tmp["label2id"]
        self.label_size=tmp["label_size"]
        self.vocabs=tmp["vocabs"]
        self.vocab2id=tmp["vocab2id"]
        self.vocab_size=tmp["vocab_size"]
        self.embedding_matrix=np.array(tmp["embedding_matrix"])
    
    # SAMPLE: docs=[{"text":["i","am","happy"],"label":{"happy":10,"sad":0,"normal":1}},...]
    def train(self,docs,model_file=None):
        self.trained=True
        docs=list(filter(lambda doc:len(doc["text"])>1,docs))
        
        if self.method=="word2vec":
            sentences=list(map(lambda x:x["text"],docs))
            if model_file:
                self.model=gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=False)
            else:
                self.model=Word2Vec()
                self.model.build_vocab(sentences)
                self.model.train(sentences,total_examples=self.model.corpus_count,epochs=self.model.iter)
            self.word_dim=self.model.vector_size
            self.vocabs=list({v for s in sentences for v in s})
            self.vocab_size=len(self.vocabs)
            self.vocab2id={self.vocabs[id]:id for id in range(self.vocab_size)}
            if "" not in self.vocabs:
                self.vocabs.append("")
                self.vocab2id[""]=self.vocab_size
                self.vocab_size+=1
            self.embedding_matrix=np.zeros((self.vocab_size, self.word_dim))
            for i in range(self.vocab_size):
                if self.vocabs[i] in self.model:
                    vec=self.model[self.vocabs[i]]
                    if vec is not None:
                        self.embedding_matrix[i]=vec
        else:
            sys.stdout.write("Invalid method!!")
        
        self.labels=[]
        for doc in docs:
            for lbl in doc["label"]:
                if not lbl in self.labels:
                    self.labels.append(lbl)
        self.label_size=len(self.labels)
        self.label2id={self.labels[id]:id for id in range(self.label_size)}

    def get_embedding(self,docs,fixed_len=None):
        if not self.trained:
            sys.stdout.write("Model not trained!!!!!")
            exit()
        docs=list(filter(lambda doc:len(doc["text"])>1,docs))
        docs=list(filter(lambda doc:doc["label"],docs))
        
        if self.method=="word2vec":
            x=[]
            for doc in docs:
                cur_sen=[]
                origin_len=len(doc["text"])
                actual_len=fixed_len if fixed_len else origin_len
                for i in range(actual_len):
                    if i<origin_len and (doc["text"][i] in self.vocab2id):
                        cur_sen.append(self.vocab2id[doc["text"][i]])
                    else:
                        cur_sen.append(self.vocab2id[""])
                x.append(cur_sen)
        else:
            sys.stdout.write("Invalid method!!")
        
        y=[]
        z=[]
        for doc in docs:
            # Deal with y
            cur_sum=0
            cur_label=[0 for i in range(self.label_size)]
            for lbl in doc["label"]:
                if lbl in self.label2id:
                    cur_sum+=doc["label"][lbl]
                    cur_label[self.label2id[lbl]]=doc["label"][lbl]
            cur_label=list(map(lambda x:x/cur_sum,cur_label))
            y.append(cur_label)
            # Deal with z
            index=0
            for j in range(1,self.label_size):
                if cur_label[j]>cur_label[index]:
                    index=j
            z.append(index)
        assert(len(x)==len(y) and len(y)==len(z))
        return x,y,z