#!/usr/bin/env python
#-*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2018 Luzzi Valerio
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#
# Name:        machine_learning.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     10/10/2018
# ------------------------------------------------------------------------------


from gecosistema_core import *

#pandas
import pandas as pd

# SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.externals import joblib

class StaticSVR(SVR):

    #SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)[source]
    def __init__(self, C = 1.0, epsilon = 0.1, gamma=0.001):
        """
        Constructor
        """
        super(StaticSVR,self).__init__(kernel='rbf', degree=3, gamma=gamma, coef0=0.0, tol=0.001, C=C, epsilon=epsilon, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        self.stdsc = StandardScaler()
        self.df =None
        self.X = None
        self.y = None


    def load(self, filecsv, sep=',', glue='"'):
        """
        load
        """
        print("loading features from %s..."%(filecsv))
        self.df  = pd.read_csv(filecsv, sep = ",", header=0, engine='c')

    def train(self, features="", target= "", dates="", train_percent=0.75):
        """
        train
        """
        features = listify(features, sep="," , glue='"')
        features = [item.encode("ascii","replace") for item in features]
        dates = dates if dates else 0
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows

        #select only feature columns
        dfX  = self.df[features]
        dfy  = self.df[target]
        dfd  = self.df[dates]

        #100% of data
        self.dates  = [strftime("%Y-%m-%d",pd.to_datetime(t).to_pydatetime()) for t in dfd.values[:]]
        self.X = dfX.values[:] #100%
        self.y = dfy.values[:] #Target 100%


        # pandas to numpy array
        if m_train >1:
            X_train = dfX.values[:m_train]
            y_train = dfy.values[:m_train]  #Train Target column

        #X_test  = dfX.values[m_train:]
        #y_test  = dfy.values[m_train:]

        #Normalization
        X_train = self.stdsc.fit_transform(X_train)
        #X_test  = self.stdsc.transform(X_test)

        #training!
        print("make SVR training(fit)...")
        self.fit(X_train, y_train)


##    def get_target(self, filecsv, sep=',', glue='"', features="", target= "", dates="", train_percent=0.75):
##        """
##        get_target
##        """
##        self.load(filecsv, sep=sep, glue=glue, features=features, target=target, dates=dates, train_percent=train_percent)
##
##        return zip(self.d_test,self.y_test)


    def prediction(self, train_percent=0.75, zipped=True):
        """
        make_prediction from csv
        """
        print("make SVR predictions...")
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows
        dates   = self.dates[m_train:]
        X_test  = self.X[m_train:]
        X_test  = self.stdsc.transform(X_test)
        y_test  = self.y[m_train:]
        predictions = self.predict(X_test)

        if zipped:
            return zip(dates,predictions,y_test)
        else:
            return (dates,predictions,y_test)


    def make_stats(self, train_percent=0.75):
        """
        make_stats from csv
        """
        print("make SVR statistics...")

        dates,s,o =  self.prediction(train_percent,False)

        self.mse = MSE(s,o)
        self.rmse =  RMSE(s,o)
        self.nash_sutcliffe = NASH(s,o)
        self.M = M(s,o)

        print ("M=%.2f MSE=%.2f RMSE=%.2f  NASH-SUTCLIFFE=%.2f"%(self.M,self.mse,self.rmse,self.nash_sutcliffe))

if __name__== "__main__":

    filecsv = r"BETANIA0.csv"
    s =StaticSVR(C=200,epsilon=0.5,gamma=0.1)

    s.load(filecsv)
    s.train(features = u"T m norm,P1,P2", target = "TARGET", dates= "DATA")
    print s.make_stats(train_percent=0)
    #print s.make_stats(filecsv,features = u"T m norm,P1,P2", target = "TARGET", dates= "DATA")