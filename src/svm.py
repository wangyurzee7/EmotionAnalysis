import os
import sys
import json
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from src.scorer import *

class Svm:
    def train_and_test(self,train_x,train_y,test_x,test_y):
        svm=SVC(kernel='linear',C=10)
        svm.fit(train_x,train_y)
        pred=svm.predict(test_x)
        print("Acc = {}".format(accuracy(pred,test_y)))
        print("Macro-F1 = {}".format(macro_f1(pred,test_y)))