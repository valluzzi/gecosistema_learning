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
        self.X_train= None
        self.y_train= None
        self.X_test = None
        self.y_test = None

    def load(self, filecsv, sep=',', glue='"', features="", target= "", dates="", train_percent=0.75):
        """
        load
        """
        print("loading features from %s..."%(filecsv))
        features = listify(features, sep=sep , glue=glue)
        features = [item.encode("ascii","replace") for item in features]

        dates = dates if dates else 0

        df = pd.read_csv(filecsv, sep = ",", header=0, engine='c')
        m,n = df.shape
        m_train = int(m*train_percent)  #number of training rows
        m_test  = m-m_train

        #select only feature columns
        dfX  = df[features]
        dfy  = df[target]
        dfd  = df[dates]

        # pandas to numpy array
        self.X_train = dfX.values[:m_train]
        self.y_train = dfy.values[:m_train]  #Train Target column
        self.d_train = dfd.values[:m_train]
        self.d_train  = [strftime("%Y-%m-%d",pd.to_datetime(t).to_pydatetime()) for t in self.d_train]

        self.X_test  = dfX.values[m_train:]
        self.y_test  = dfy.values[m_train:]
        self.d_test  = dfd.values[m_train:]
        self.d_test  = [strftime("%Y-%m-%d",pd.to_datetime(t).to_pydatetime()) for t in self.d_test]
        #Normalization
        self.X_train = self.stdsc.fit_transform(self.X_train)
        self.X_test  = self.stdsc.transform(self.X_test)

    def get_target(self, filecsv, sep=',', glue='"', features="", target= "", dates="", train_percent=0.75):
        """
        get_target
        """
        if self.X_train is None:
            self.load(filecsv, sep=sep, glue=glue, features=features, target=target, dates=dates, train_percent=train_percent)

        return zip(self.d_test,self.y_test)


    def train(self, filecsv, sep=',', glue='"', features="", target= "", dates="", train_percent=0.75):
        """
        train fron csv
        """

        if self.X_train is None:
            self.load(filecsv, sep=sep, glue=glue, features=features, target=target, dates=dates, train_percent=train_percent)

        #training!
        print("training the SVR...")
        self.fit(self.X_train, self.y_train)


    def make_prediction(self, filecsv, sep=',', glue='"', features="", target= "",  dates="", train_percent=0.75):
        """
        make_prediction from csv
        """
        if self.X_train is None:
            self.train(filecsv, sep=sep, glue=glue, features=features, target=target, dates=dates, train_percent=train_percent)

        print("make predictions...")
        predictions = self.predict(self.X_test)
        return zip(self.d_test,predictions)



if __name__== "__main__":

    filecsv = r"BETANIA0.csv"
    s =StaticSVR()
    #s.train(filecsv,features = u"T m norm,P1,P2", target = "TARGET")
    print s.make_prediction(filecsv,features = u"T m norm,P1,P2", target = "TARGET", dates= "DATA")
    print s.make_prediction(filecsv,features = u"T m norm,P1,P2", target = "TARGET", dates= "DATA")