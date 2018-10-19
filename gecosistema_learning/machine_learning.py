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
import matplotlib.pyplot as plt
# SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.externals import joblib

class StaticSVR(SVR):

    #SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)[source]
    def __init__(self, C = 1.0, epsilon = 0.1, gamma=0.001, filecsv=None):
        """
        Constructor
        """
        super(StaticSVR,self).__init__(kernel='rbf', degree=3, gamma=gamma, coef0=0.0, tol=0.001, C=C, epsilon=epsilon, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        self.stdsc = StandardScaler()
        self.df =None
        self.X = None
        self.y = None
        self.train_percent = 0.75
        self.predictions =None
        if isfile(filecsv):
            load(filecsv)

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
        self.train_percent = train_percent
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

    def prediction(self, train_percent=0.75, zipped=False):
        """
        make_prediction from csv
        """
        print("make SVR predictions...")
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows
        dates   = self.dates[m_train:]
        X_test  = self.X
        X_test  = self.stdsc.transform(X_test)
        self.predictions = self.predict(X_test) #predict on all the domain

        res =  (dates[m_train:],self.predictions[m_train:],self.y[m_train:])  #show only test percent
        return zip(res) if zipped else res

    def target(self, train_percent=0.0, zipped=False):
        """
        target - return the target array
        """
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows
        y_test  = self.y[m_train:]
        dates   = self.dates[m_train:]
        res = dates,y_test
        return zip(res) if zipped else res

    def make_stats(self, train_percent=0.75):
        """
        make_stats from csv
        """
        print("make SVR statistics...")
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows

        #dates,s,o =  self.prediction(train_percent,False)
        if self.predictions is None:
            dates,s,o =  self.prediction(train_percent,False)
        else:
            s = self.predictions[m_train:]
            o = self.y[m_train:]

        self.mse = MSE(s,o)
        self.rmse =  RMSE(s,o)
        self.nash_sutcliffe = NASH(s,o)
        self.M = M(s,o)

        print ("M=%.2f MSE=%.2f RMSE=%.2f  NASH-SUTCLIFFE=%.2f"%(self.M,self.mse,self.rmse,self.nash_sutcliffe))

    def plot(self, train_percent=0.75):
        """
        plot predictions
        """
        m,n = self.df.shape
        m_train = int(m*train_percent)  #number of training rows
        s       = self.predictions[m_train:]
        s_train = self.predictions[:m_train]
        dates = self.dates
        o = self.y

        plt.plot(dates,o,dates[:m_train],s_train,dates[m_train:],s)
        plt.show()


if __name__== "__main__":

    filecsv = r"BETANIA0.csv"

    svr =StaticSVR(C=1e6,epsilon=1e2,gamma=0.1)

    svr.load(filecsv)
    svr.train(features = u"T-12,T0-1,T m norm,P1,P2,P7,P8,P9,P10,P11,P14,P17,T1 (K),T2 (K),T7 (K),T8 (K),T9 (K),T10 (K),T11 (K),T14 (K),T17 (K)", target = "TARGET", dates= "DATA", train_percent=0.75)
    print svr.make_stats(train_percent=0.75)


    print svr.plot()
