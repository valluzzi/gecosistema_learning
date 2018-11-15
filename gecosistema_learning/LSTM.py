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
import pandas as pd
import matplotlib.pyplot as plt

#keras
from keras.models import Sequential
from keras.layers import *

class SimpleLSTM(Sequential):

    def __init__( neurons , m, n, dropout =0.2, dense=1 ):
        super(Sequential,self)
        self.add(LSTM(neurons, input_shape=(m,n)))
        self.add(Dropout(dropout))
        self.add(Dense(dense))
        self.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

        self.stdsc = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.train_percent = 0.75
        self.predictions = None


    def load(self,filename):
        """
        load
        """
        self.df = pd.read_csv(filecsv, sep=",", header=0, comment="#", engine='c')

    def train(self, features="", droplist="", target= "", dates="", train_percent=0.75):
        """
        train
        """
        features = listify(features, sep=",", glue='"')
        features = [item.encode("ascii", "replace") for item in features]
        droplist = listify(droplist, sep=",", glue='"')
        droplist = [item.encode("ascii", "replace") for item in droplist]

        features = features if features else list(self.df.columns.values)
        features = [feature for feature in features if feature not in (target, dates)]

        # remove fields in droplist
        features = features if not droplist else [feature for feature in features if feature not in droplist]

        dates = dates if dates else 0
        self.train_percent = train_percent
        m, n = self.df.shape
        m_train = int(m * train_percent)  # number of training rows

        # select only feature columns
        dfX = self.df[features]
        dfy = self.df[target]
        dfd = self.df[dates]

        # 100% of data
        self.dates = [strftime("%Y-%m-%d", pd.to_datetime(t).to_pydatetime()) for t in dfd.values[:]]
        self.X = dfX.values[:]  # 100%
        self.y = dfy.values[:]  # Target 100%

        # pandas to numpy array
        if m_train > 1:
            X_train = dfX.values[:m_train]
            y_train = dfy.values[:m_train]  # Train Target column
            X_test  = dfx.values[m_train:]
            y_test  = dfy.values[m_train:]
        # Normalization
        X_train = self.stdsc.fit_transform(X_train)

        # training!
        print("make LSTM training(fit)...")
        self.fit(X_train, y_train,epochs=1500, batch_size=5, validation_data=(X_test, y_test), verbose=2, shuffle=False)

    def prediction(self, train_percent=0.75, zipped=False):
        pass

if __name__== "__main__":

    filecsv = r"BETANIA5int.csv"

    m,n = 288, 11
    svr = SimpleLSTM(3,m,n)

    svr.load(filecsv)
    #svr.train(droplist = "P1,P2,P9,P8,T1,T7,T8,T9,T11,T14,T17,E1", target = "TARGET", dates= "DATA", train_percent=0.75)

    #print svr.make_stats(train_percent=0.75)