#coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np;
import os
import json
import random
from src.scorer import *
try:
    from progressbar import ProgressBar
    USING_BAR=True
except:
    USING_BAR=False

class Cnn(nn.Module):
    def __init__(self,args):
        super(Cnn,self).__init__()
        self.fixed_len=args['fixed_len']
        self.word_dim=args['word_dim']
        
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        self.conv1=nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv2=nn.Sequential(
                    nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv3=nn.Sequential(
                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv4=nn.Sequential(
                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        final_n=args['fixed_len']//16
        final_m=args['word_dim']//16
        self.out=nn.Linear(final_n*final_m*128,args['label_size'])
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(x.size(0),1,self.fixed_len,self.word_dim)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.size(0),-1)
        ret=self.out(x)
        return ret


class Mlp(nn.Module):
    def __init__(self,args):
        super(Mlp,self).__init__()
        self.fixed_len=args['fixed_len']
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        input_size=args['fixed_len']*args['word_dim']
        self.linear=nn.Sequential(
                    nn.Linear(input_size,input_size//4),
                    nn.ReLU(inplace=True),
                    nn.Linear(input_size//4,args['label_size'])
                    )
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x

class Classifier:
    # `cnn_args` should countain key: fixed_len,vocab_size,word_dim,label_size,embedding_matrix
    def __init__(self,cnn_args,LR=0.001,epoch_size=4,regression=True,network="mlp"):
        self.net_name=network
        self.regression=regression
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if network=="cnn":
            self.model=nn.DataParallel(Cnn(cnn_args))
        elif network=="mlp":
            self.model=nn.DataParallel(Mlp(cnn_args))
        self.model.to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=LR)
        if regression:
            self.loss_function=nn.MSELoss()
        else:
            self.loss_function=nn.CrossEntropyLoss()
        self.epoch_size=epoch_size
    def train(self,x,y):
        running_loss=0.0
        running_acc=0.0
        n=len(x)
        epoch_size=self.epoch_size
        
        _r=range(int(n/epoch_size))
        if USING_BAR:
            _r=ProgressBar()(_r)
        for i in _r:
            l,r=i*epoch_size,min(i*epoch_size+epoch_size,n)
            batch_x=Variable(torch.LongTensor(x[l:r])).to(self.device)
            if self.regression:
                batch_y=Variable(torch.Tensor(y[l:r])).to(self.device)
            else:
                batch_y=Variable(torch.LongTensor(y[l:r])).to(self.device)
            self.optimizer.zero_grad()
            output=self.model(batch_x)
            loss=self.loss_function(output,batch_y)
            loss.backward()
            self.optimizer.step()
            running_loss+=loss.item()
            # Calculate accuracy
            pred_y=torch.max(output,1)[1].data.squeeze()
            if self.regression:
                result_y=torch.max(batch_y,1)[1].data.squeeze()
            else:
                result_y=batch_y
            acc=(result_y==pred_y)
            acc=acc.cpu().numpy().sum()
            running_acc+=acc
        return {"loss":running_loss/(i+1),"accuracy":running_acc/n}
    def test(self,x):
        ret_y=[]
        ret_z=[]
        n=len(x)
        epoch_size=self.epoch_size
        
        _r=range(int(n/epoch_size))
        if USING_BAR:
            _r=ProgressBar()(_r)
        for i in _r:
            l,r=i*epoch_size,min(i*epoch_size+epoch_size,n)
            batch_x=Variable(torch.LongTensor(x[l:r])).to(self.device)
            output=self.model(batch_x)
            ret_y.extend(output.cpu().detach().numpy().tolist()) # Deal with output (regression)
            # Deal with output (Classifier only)
            pred_y=torch.max(output,1)[1].data.squeeze()
            ret_z.extend(pred_y.cpu().numpy().tolist())
        return ret_y,ret_z
    def train_and_test(self,train_x,train_y,test_x,test_y,test_z,epoch=3):
        for i in range(epoch):
            print("[ Epoch#{} ]".format(i+1))
            info=self.train(train_x,train_y)
            print(info)
            pred_y,pred_z=self.test(test_x)
            print("Acc = {}".format(accuracy(pred_z,test_z)))
            print("F-Score = {}".format(f_score(pred_z,test_z)))
            print("Coef = {}".format(corr(pred_y,test_y)))
            print("")
        return self.test(test_x)
