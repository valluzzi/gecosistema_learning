# -------------------------------------------------------------------------------
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
# Name:        LSTM.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     05/11/2018
# -------------------------------------------------------------------------------
from gecosistema_core import *

#import tensorflow
#pandas
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

#keras
from keras.models import Sequential
from keras.layers import *

class SimpleLSTM(Sequential):

    def __init__(self, neurons=3 , dropout =0.05, dense=1, train_percent = 0.75 ):
        Sequential.__init__(self)

        self.neurons = neurons
        self.dropout = dropout
        self.dense=dense

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df = None
        self.X = None
        self.y = None
        self.train_percent = train_percent
        self.predictions = None


    def load(self,filename):
        """
        load
        """
        self.df = pd.read_csv(filename, sep=",", header=0, comment="#", engine='c')

    def T_minus(self,n):
        """
        T_minus
        """


    def train(self, features="", droplist="", target= "", dates="",  epochs = 600):
        """
        train
        """
        features = listify(features, sep=",", glue='"')
        droplist = listify(droplist, sep=",", glue='"')

        features = features if features else list(self.df.columns.values)
        features = [feature for feature in features if feature not in (target, dates)]

        # remove fields in droplist
        features = features if not droplist else [feature for feature in features if feature not in droplist]

        dates = dates if dates else 0
        m, n = self.df.shape
        m_train = int(m * self.train_percent)  # number of training rows

        # Normalization
        scaled = self.scaler.fit_transform(self.df[[target]+features].values[:])

        #train -test split
        dfXs = scaled[:, 1:]
        dfys = scaled[:, 0]

        #add T-6
        T   = dfys
        T_6 = T[:-6]
        m_new = T_6.shape[0]
        m_train = int(m_new * self.train_percent)  # number of training rows
        dfys = dfys[6:]
        dfXs = dfXs[6:,:]
        dfXs = np.hstack((T_6.reshape((m_new,1)),dfXs))

        # select only feature columns
        dfd = self.df[dates]

        if m_train > 1:
            X_train = dfXs[:m_train]
            y_train = dfys[:m_train]  # Train Target column
            X_test  = dfXs[m_train:]
            y_test  = dfys[m_train:]

        # 100% of data
        self.dates = [strftime("%Y-%m-%d", pd.to_datetime(t).to_pydatetime()) for t in dfd.values[:]]
        self.X = dfXs # 100%
        self.y = self.df[target].values[6:]  # Target 100%

        #reshape
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test  = X_test.reshape((  X_test.shape[0], 1, X_test.shape[1]))

        # training!
        print("make LSTM training(fit)...")

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        #
        self.add(LSTM(self.neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.add(Dropout(self.dropout))
        self.add(Dense(self.dense))
        self.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.fit(X_train, y_train, epochs=epochs, batch_size=1, validation_data=(X_test, y_test), verbose=0, shuffle=False)

    def prediction(self, train_percent=-1, zipped=False):
        """
        make_prediction
        """
        print("make LSTM predictions...")
        m, _ = self.X.shape
        train_percent = train_percent if train_percent >= 0 else self.train_percent
        m_train = int(m *train_percent)  # number of training rows

        X_test = self.X
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        self.predictions = self.predict(X_test)  # predict on all the domain
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
        inv_yhat = np.concatenate((self.predictions, X_test[:,1:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        self.predictions = inv_yhat[:, 0]

        if zipped:
            return list(zip(self.dates[m_train:], self.predictions[m_train:], self.y[m_train:]))
        else:
            return (self.dates[m_train:], self.predictions[m_train:], self.y[m_train:])

    def make_stats(self, train_percent=-1):
        """
        make_stats
        """
        print("make SVR statistics...")
        m,_= self.X.shape
        train_percent = train_percent if train_percent>0 else self.train_percent
        m_train = int(m* train_percent)  #number of training rows

        if self.predictions is None:
            dates,s,o =  self.prediction(train_percent, zipped = False)
        else:
            s = self.predictions[m_train:]
            o = self.y[m_train:]

        self.mse = MSE(s,o)
        self.rmse =RMSE(s,o)
        self.nash_sutcliffe = NASH(s,o)
        self.M = M(s,o)

        print ("M=%.2f MSE=%.2f RMSE=%.2f  NASH-SUTCLIFFE=%.3f"%(self.M,self.mse,self.rmse,self.nash_sutcliffe))


if __name__== "__main__":

    filecsv = r"BETANIA5int.csv"

    svr = SimpleLSTM(train_percent = 0.745)

    svr.load(filecsv)
    svr.train(droplist = "P1,P2,P9,P8,T1,T7,T8,T9,T11,T14,T17,E1", target = "TARGET", dates= "DATA",  epochs=6)

    print(svr.prediction(train_percent=0.0,zipped=True))
    #svr.make_stats(1train_percent =0.5)