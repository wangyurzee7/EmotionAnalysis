#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _p(buf):
    print(buf)
    with open("result.log.txt","a") as f:
        f.write(str(buf)+'\n')

class Cnn(nn.Module):
    def __init__(self,args):
        super(Cnn,self).__init__()
        self.fixed_len=args['fixed_len']
        self.word_dim=args['word_dim']
        
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        self.conv1=nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv2=nn.Sequential(
                    nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv3=nn.Sequential(
                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv4=nn.Sequential(
                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        final_n=args['fixed_len']//16
        final_m=args['word_dim']//16
        self.fc=nn.Linear(final_n*final_m*128,args['label_size'])
        # self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(x.size(0),1,self.fixed_len,self.word_dim)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        # x=self.softmax(x)
        return x

class TextCnn(nn.Module):
    def __init__(self,args):
        super(TextCnn,self).__init__()
        self.fixed_len=args['fixed_len']
        self.word_dim=args['word_dim']
        
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        
        kernels=[2,3,4,5]
        oc=16
        self.convs=nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=oc,kernel_size=(k,self.word_dim)) for k in kernels])
        
        # self.dropout=nn.Dropout(0.5)
        self.fc=nn.Linear(oc*len(kernels),args['label_size'])
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(x.size(0),-1,self.fixed_len,self.word_dim) # [batch,1,len,dim]
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs] # [[batch,16,len],...]
        x=[F.max_pool1d(c,int(c.size(2))).squeeze(2) for c in x] # [[batch,16],...]
        x=torch.cat(x,1)
        # x=self.dropout(x)
        x=self.fc(x)
        x=self.softmax(x)
        return x

class Mlp(nn.Module):
    def __init__(self,args):
        super(Mlp,self).__init__()
        self.fixed_len=args['fixed_len']
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        input_size=args['fixed_len']*args['word_dim']
        self.linear=nn.Sequential(
                    nn.Linear(input_size,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64,args['label_size'])
                    )
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        x=self.softmax(x)
        return x

class Rnn(nn.Module):
    def __init__(self,args,using_gru=False):
        super(Rnn,self).__init__()
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        
        self.hidden_size=128
        self.n_layers=4
        
        if using_gru:
            self.rnn=nn.GRU(input_size=args['word_dim'],hidden_size=self.hidden_size,num_layers=self.n_layers)
        else:
            self.rnn=nn.RNN(input_size=args['word_dim'],hidden_size=self.hidden_size,num_layers=self.n_layers)
        
        self.fc=nn.Linear(self.hidden_size,args['label_size'])
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x,hid):
        x=self.embeding(x)
        x=x.view(x.size(0),-1,self.word_dim)
        x,hid=self.rnn(x,hid)
        x=x[:,-1,:]
        x=self.fc(x)
        # print("After fc")
        # print(x)
        x=self.softmax(x)
        # print("After softmax")
        # print(x)
        return x
    def initial_hid(self,length):
        return torch.autograd.Variable(torch.zeros(self.n_layers,length,self.hidden_size))

class Lstm(nn.Module):
    def __init__(self,args):
        super(Lstm,self).__init__()
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        
        self.hidden_size=128
        self.n_layers=8
        
        self.lstm=nn.LSTM(input_size=args['word_dim'],hidden_size=self.hidden_size,num_layers=self.n_layers,bidirectional=True)
        
        self.out=nn.Sequential(nn.Linear(self.hidden_size,args['label_size']),nn.Softmax(dim=1))
        # self.softmax=nn.Softmax(dim=1)
    def forward(self,x,hid):
        x=self.embeding(x)
        x=x.view(x.size(0),-1,self.word_dim)
        x,hid=self.lstm(x)
        x=x[:,-1,:]
        x=self.out(x)
        return x
    def initial_hid(self,length):
        return torch.autograd.Variable(torch.zeros(self.n_layers,length,self.hidden_size))

class Rcnn(nn.Module):
    def __init__(self,args,using_gru=False):
        super(Rcnn,self).__init__()
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        
        self.hidden_size=128
        self.n_layers=4
        self.dropout=0.1
        
        self.lstm=nn.LSTM(input_size=args['word_dim'],hidden_size=self.hidden_size,num_layers=self.n_layers,batch_first=True,bidirectional=True,dropout=self.dropout)
        self.fc1=nn.Linear(2*self.hidden_size+self.word_dim,self.hidden_size)
        
        self.fc2=nn.Linear(self.hidden_size,args['label_size'])
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.embeding(x)
        x=x.view(1,-1,self.word_dim) # [1,len,word_dim)
        y,_=self.lstm(x,None) # [1,len,2*hidden_size]
        x=torch.cat([x,y],2)
        x=self.fc1(x) # [1,len,hidden_size]
        x=x.permute(0,2,1) # [1,hidden_size,len]
        x=F.max_pool1d(x,int(x.size()[2])) # [1,hidden_size,1]
        x=x.squeeze(2)
        x=self.fc2(x)
        x=self.softmax(x)
        return x

class Clstm(nn.Module):
    def __init__(self,args):
        super(Clstm,self).__init__()
        self.word_dim=args['word_dim']
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        
        kernels=[2,4,6,8]
        oc=16
        self.hidden_size=len(kernels)*oc
        self.dropout=0.5
        self.embeding=nn.Embedding(args['vocab_size'],args['word_dim'],_weight=torch.Tensor(args['embedding_matrix']))
        # CNN
        self.convs=nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=oc,kernel_size=(k,self.word_dim),stride=1,padding=(k//2,0)) for k in kernels])
        # LSTM
        self.n_layers=8
        self.lstm=nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers= self.n_layers,dropout=self.dropout,bidirectional=True)
        # linear
        self.fc=nn.Linear(self.hidden_size*2,args['label_size'])
        # dropout
        self.dropout=nn.Dropout(self.dropout)

    def forward(self, x):
        x=self.embeding(x)
        # CNN
        x=self.dropout(x)
        x=x.unsqueeze(1)
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [[batch,16,len],...]
        x=torch.cat(x,1)
        x=torch.transpose(x,1,2)
        # LSTM
        x,_= self.lstm(x)
        # operation1
        # x=torch.transpose(x, 1, 2)
        # x=F.max_pool1d(x, x.size(2)).squeeze(2)
        # operation2
        x=x[:,-1,:]
        # linear
        x=self.fc(torch.tanh(x))
        return x

class Classifier:
    # `cnn_args` should countain key: fixed_len,vocab_size,word_dim,label_size,embedding_matrix
    def __init__(self,net_args,LR=0.001,batch_size=4,regression=True,network="mlp"):
        self.net_name=network
        self.regression=regression
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if network=="cnn":
            self.model=nn.DataParallel(Cnn(net_args))
            self.has_hid=False
        if network=="textcnn":
            self.model=nn.DataParallel(TextCnn(net_args))
            self.has_hid=False
        elif network=="mlp":
            self.model=nn.DataParallel(Mlp(net_args))
            self.has_hid=False
        elif network=="rnn":
            self.model=Rnn(net_args)
            self.model=nn.DataParallel(self.model)
            self.has_hid=True
            assert(batch_size==1)
        elif network=="gru":
            self.model=Rnn(net_args,using_gru=True)
            self.model=nn.DataParallel(self.model)
            self.has_hid=True
            assert(batch_size==1)
        elif network=="lstm":
            self.model=Lstm(net_args)
            self.model=nn.DataParallel(self.model)
            self.has_hid=True
            assert(batch_size==1)
        elif network=="rcnn":
            self.model=nn.DataParallel(Rcnn(net_args))
            self.has_hid=False
        elif network=="clstm":
            self.model=nn.DataParallel(Clstm(net_args))
            self.has_hid=False
        self.model.to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=LR)
        if regression:
            self.loss_function=nn.MSELoss()
        else:
            self.loss_function=nn.CrossEntropyLoss()
        self.batch_size=batch_size
    def train(self,x,y):
        running_loss=0.0
        running_acc=0.0
        n=len(x)
        batch_size=self.batch_size
        
        _r=range((n-1)//batch_size+1)
        if USING_BAR:
            _r=ProgressBar()(_r)
        for i in _r:
            l,r=i*batch_size,min((i+1)*batch_size,n)
            batch_x=Variable(torch.LongTensor(x[l:r])).to(self.device)
            if self.regression:
                batch_y=Variable(torch.Tensor(y[l:r])).to(self.device)
            else:
                batch_y=Variable(torch.LongTensor(y[l:r])).to(self.device)
            self.optimizer.zero_grad()
            if self.has_hid:
                output=self.model(batch_x,self.model.module.initial_hid(batch_x.size(1)).to(self.device))
            else:
                output=self.model(batch_x)
            loss=self.loss_function(output,batch_y)
            loss.backward()
            self.optimizer.step()
            running_loss+=loss.item()
            '''
            print(output)
            print(batch_y)
            print(loss)
            '''
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
        batch_size=self.batch_size
        
        _r=range((n-1)//batch_size+1)
        if USING_BAR:
            _r=ProgressBar()(_r)
        for i in _r:
            l,r=i*batch_size,min((i+1)*batch_size,n)
            batch_x=Variable(torch.LongTensor(x[l:r])).to(self.device)
            if self.has_hid:
                output=self.model(batch_x,self.model.module.initial_hid(batch_x.size(1)).to(self.device))
            else:
                output=self.model(batch_x)
            ret_y.extend(output.cpu().detach().numpy().tolist()) # Deal with output (regression)
            # Deal with output (Classifier only)
            pred_y=torch.max(output,1)[1].data.squeeze()
            pred_y_list=pred_y.cpu().numpy().tolist()
            try:
                ret_z.extend(pred_y_list)
            except:
                ret_z.append(pred_y_list)
        return ret_y,ret_z
    def draw(self,x):
        from tensorboardX import SummaryWriter
        with SummaryWriter(comment=self.net_name) as w:
            batch_x=Variable(torch.LongTensor(x[0:1])).to(self.device)
            if self.has_hid:
                hid=self.model.module.initial_hid(batch_x.size(1)).to(self.device)
                w.add_graph(self.model, (batch_x,hid, ))
            else:
                w.add_graph(self.model, (batch_x, ))
    def train_and_test(self,train_x,train_y,test_x,test_y,test_z,epoch=3):
        for i in range(epoch):
            _p("[ Epoch#{} ]".format(i+1))
            info=self.train(train_x,train_y)
            _p(info)
            pred_y,pred_z=self.test(test_x)
            _p("Acc = {}".format(accuracy(pred_z,test_y)))
            _p("Macro-F1 = {}".format(macro_f1(pred_z,test_y)))
            _p("Coef = {}".format(corr(pred_y,test_y)))
            _p("")
        return self.test(test_x)
