# Databricks notebook source
# Authors:
#     1. PALLAVI DASARAPU - PXD210008
#     2. TIRTH MEHTA - TDM210001
#     3. ANKIT SAHU - AXS210226

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

pip install keras

# COMMAND ----------

pip install yfinance

# COMMAND ----------

pip install pandas_datareader

# COMMAND ----------

from keras.layers import Dense,LSTM
from keras import backend as bknd
from keras.models import Sequential
from keras import metrics as mtrs
import numpy as np
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# COMMAND ----------

import yfinance as yfin
from pandas_datareader import data as pdr
yfin.pdr_override()

# COMMAND ----------

strt_dt = '2012-01-01'
ed_dt = '2023-03-31'

# COMMAND ----------

tsldatadf = pdr.get_data_yahoo('TSLA', start= strt_dt, end= ed_dt)
print(type(tsldatadf))

# COMMAND ----------

tsldatadf['Date'] = tsldatadf.index
tsl_data_df = spark.createDataFrame(tsldatadf)
print(type(tsl_data_df))

# COMMAND ----------

tsldatadf.display()

# COMMAND ----------

datetmp = tsl_data_df.select('Date')
clostmp = tsl_data_df.select('Close')

# COMMAND ----------

plt.title('Stock input data - TSLA')
x = datetmp.collect()
y = clostmp.collect()
plt.legend(['Stock Price Data - TSLA'])
plt.ylabel('StockPrice')
plt.xlabel('Year')
plt.plot(x,y)

# COMMAND ----------

# data from close
closdatadf = clostmp.collect()

# COMMAND ----------

# scaling (0-1)
from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

mnmxscl = MinMaxScaler(feature_range=(0,1))
closscaled=mnmxscl.fit_transform(closdatadf)

# COMMAND ----------

#creating training data
tmp = len(closdatadf)
trndatasize=int(tmp*0.8)
trndatasize

# COMMAND ----------

trndf=closscaled[0:trndatasize]
x_trnvls=[]
y_trnvls=[]
hops=50
diff = trndatasize - hops
for xx in range(diff):
    x_trnvls.append(trndf[xx : xx+hops, 0])
    y_trnvls.append(trndf[xx+hops, 0])

# COMMAND ----------

# Converting x_trnvls and y_trnvls values to numpy arrays
xtr = np.array(x_trnvls)
ytr = np.array(y_trnvls)
x_trnvls,y_trnvls = xtr, ytr
x_trnvls.shape
 

# COMMAND ----------

#Reshaping data
x_trnvls = np.reshape(x_trnvls,(diff,hops,1))
x_trnvls.shape

# COMMAND ----------

valLstm = Sequential()
valLstm.add(LSTM(units=50, input_shape=(hops,1), return_sequences=True))
valLstm.add(LSTM(units=50,return_sequences=False))

# COMMAND ----------

dns = [10, 25]
valLstm.add(Dense(dns[1]))
valLstm.add(Dense(dns[0]))

# COMMAND ----------

def rtmnsqr(act, prd):
    bcksqr = bknd.square(prd - act)
    bckmean = bknd.mean(bcksqr, axis=-1)
    return bknd.sqrt(bckmean)

# COMMAND ----------

msqerr = mtrs.mean_squared_error
mabserr = mtrs.mean_absolute_error
mabsprcnt = mtrs.mean_absolute_percentage_error
valLstm.compile(optimizer='adam', metrics=[msqerr, mabserr, mabsprcnt, rtmnsqr], loss='mean_squared_error')

# COMMAND ----------

btchsz = [1,5,10]
epo = 50
btch1 = valLstm.fit(x_trnvls, y_trnvls, batch_size=btchsz[0], epochs=epo)
btch2 = valLstm.fit(x_trnvls, y_trnvls, batch_size=btchsz[1], epochs=epo)
btch3 = valLstm.fit(x_trnvls, y_trnvls, batch_size=btchsz[2], epochs=epo)

# COMMAND ----------

#Testing data set
print(hops)
testing_data = closscaled[diff:]
x_tstvls = []
y_tstvls = closdatadf[trndatasize:]
val = len(testing_data)-hops
for x in range(val):
    x_tstvls.append(testing_data[x:x+hops,0])

# COMMAND ----------

# Converting x_trnvls and y_trnvls into numpy arrays
x_tstvls = np.array(x_tstvls)
x_tstvls.shape

# COMMAND ----------

#Reshaping data
x_tstvls = np.reshape(x_tstvls,(val,hops,1))

# COMMAND ----------

#predicting values
prdvals = valLstm.predict(x_tstvls)
prdvals = mnmxscl.inverse_transform(prdvals)

# COMMAND ----------

def rtmnsqr_func(m,n):
    temp = (m-n)**2
    temp = np.mean(temp)
    temp = np.sqrt(temp)
    return temp

# COMMAND ----------

x = np.mean((prdvals-y_tstvls)**2)
print(x)

# COMMAND ----------

print(np.mean(mtrs.mean_absolute_percentage_error(y_tstvls, prdvals)))

# COMMAND ----------

rtmnsqr_val = rtmnsqr_func(prdvals, y_tstvls)
print("Root mean square Error for the prediction:", rtmnsqr_val)

# COMMAND ----------

yrval = datetmp.collect()
closval = clostmp.collect()
plt.title('LSTM - Training Data Vs Validation Data Vs Prediction Data')
plt.legend(['Training','Validation','Prediction'])
plt.ylabel('StockPrice')
plt.xlabel('Year')
plt.plot(yrval[:trndatasize],closval[:trndatasize])
plt.plot(yrval[trndatasize:],closval[trndatasize:])
plt.plot(yrval[trndatasize:],prdvals)

# COMMAND ----------

plt.plot(btch1.history['rtmnsqr'])
plt.plot(btch2.history['rtmnsqr'])
plt.plot(btch3.history['rtmnsqr'])
plt.ylabel('RootMeanSqrErr')
plt.xlabel('Epochs')
plt.title('Batchwise RootMeanSqr')
plt.legend(['batchsize = 1','batchsize = 5','batchsize = 10'])
plt.show()

# COMMAND ----------

siz = len(x_trnvls)
knvldsz = int(siz*0.2)
kntrnsz = int(siz*0.8)
kntstsz = int(len(closdatadf)*0.2)

# COMMAND ----------

xknvld = x_trnvls[:knvldsz]
yknvld = y_trnvls[:knvldsz]
xkntrn = x_trnvls[knvldsz:]
ykntrn = y_trnvls[knvldsz:]

# COMMAND ----------

xlen = len(xknvld)
x1knvld = [0]*xlen
for xx in range(0, xlen):
    for xy in range(0, len(xknvld[xx])):
        x1knvld[xx] += xknvld[xx][xy][0]
    x1knvld[xx] /= len(xknvld[xx])
    
y1knvld = yknvld

# COMMAND ----------

xlen = len(xkntrn)
x1kntrn = [0]*xlen
nlen = len(xkntrn[0])
for xx in range(xlen):
    for xy in range(nlen):
        x1kntrn[xx] += xkntrn[xx][xy][0]
    x1kntrn[xx] /= len(xkntrn[xx])
    
y1kntrn = ykntrn

# COMMAND ----------

xlen = len(x1kntrn)
kntrn = [0]*xlen
for pos in range(xlen):
    kntrn[pos] = [x1kntrn[pos],y1kntrn[pos]]


# COMMAND ----------

def euclDist(m, n):
    dist = 0
    dist += (m-n)**2
    dist = dist**(1/2)
    return dist

# COMMAND ----------

def knFunc(trnDt, tstVl, x):
    nbrs = []
    for i, trnVal in enumerate(trnDt):
        dist = euclDist(trnVal[:-1], tstVl)
        nbrs.append((dist, i))
        
    sort_nbrs = sorted(nbrs)
    knnbrs = sort_nbrs[:x]
    knvals = [trnDt[x][-1] for dist, x in knnbrs]
    
    tstprds = sum(knvals)/len(knvals)
    return tstprds

# COMMAND ----------

kval = [1,3,5,10,25,45,70,85,100]
 
vldknk1 = []
vldknk2 = []
vldknk3 = []
vldknk4 = []
vldknk5 = []
vldknk6 = []
vldknk7 = []
vldknk8 = []
vldknk9 = []

# COMMAND ----------

for i in x1knvld:
    vldknk1.append(knFunc(kntrn, i, kval[0]))
    vldknk2.append(knFunc(kntrn, i, kval[1]))
    vldknk3.append(knFunc(kntrn, i, kval[2]))
    vldknk4.append(knFunc(kntrn, i, kval[3]))
    vldknk5.append(knFunc(kntrn, i, kval[4]))
    vldknk6.append(knFunc(kntrn, i, kval[5]))
    vldknk7.append(knFunc(kntrn, i, kval[6]))
    vldknk8.append(knFunc(kntrn, i, kval[7]))
    vldknk9.append(knFunc(kntrn, i, kval[8]))

# COMMAND ----------

knvldpredk1 = []
knvldpredk2 = []
knvldpredk3 = []
knvldpredk4 = []
knvldpredk5 = []
knvldpredk6 = []
knvldpredk7 = []
knvldpredk8 = []
knvldpredk9 = []

# COMMAND ----------

klen = len(vldknk1)
for x in range(0, klen):
    knvldpredk1.append([vldknk1[x]])
    knvldpredk2.append([vldknk2[x]])
    knvldpredk3.append([vldknk3[x]])
    knvldpredk4.append([vldknk4[x]])
    knvldpredk5.append([vldknk5[x]])
    knvldpredk6.append([vldknk6[x]])
    knvldpredk7.append([vldknk7[x]])
    knvldpredk8.append([vldknk8[x]])
    knvldpredk9.append([vldknk9[x]])

# COMMAND ----------

def invrs_func(val):
    return mnmxscl.inverse_transform(val)

# COMMAND ----------

knvldpredk1 = invrs_func(knvldpredk1)
knvldpredk2 = invrs_func(knvldpredk2)
knvldpredk3 = invrs_func(knvldpredk3)
knvldpredk4 = invrs_func(knvldpredk4)
knvldpredk5 = invrs_func(knvldpredk5)
knvldpredk6 = invrs_func(knvldpredk6)
knvldpredk7 = invrs_func(knvldpredk7)
knvldpredk8 = invrs_func(knvldpredk8)
knvldpredk9 = invrs_func(knvldpredk9)

# COMMAND ----------

knvldact = []
ylen = len(y1knvld)
for pos in range(ylen):
    knvldact.append([y1knvld[pos]])
    
knvldact = invrs_func(knvldact)

# COMMAND ----------

rtmnsqrk = []
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk1))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk2))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk3))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk4))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk5))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk6))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk7))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk8))
rtmnsqrk.append(rtmnsqr_func(knvldact,knvldpredk9))
 
print("\n",
      "The RootMeanSqr when K is 1   : ", rtmnsqrk[0], "\n",
      "The RootMeanSqr when K is 3   : ", rtmnsqrk[1], "\n",
      "The RootMeanSqr when K is 5   : ", rtmnsqrk[2], "\n",
      "The RootMeanSqr when K is 10  : ", rtmnsqrk[3], "\n",
      "The RootMeanSqr when K is 25  : ", rtmnsqrk[4], "\n",
      "The RootMeanSqr when K is 45  : ", rtmnsqrk[5], "\n",
      "The RootMeanSqr when K is 70  : ", rtmnsqrk[6], "\n",
      "The RootMeanSqr when K is 85  : ", rtmnsqrk[7], "\n",
      "The RootMeanSqr when K is 100 : ", rtmnsqrk[8], "\n")

# COMMAND ----------

plt.ylabel('RootMeanSquare')
plt.xlabel('K_value')
plt.title('Kval Vs RootMeanSqr')
plt.plot(kval, rtmnsqrk, marker ='o')

# COMMAND ----------

xlen = len(x_tstvls)
xkntst = [0]*xlen
nlen = len(x_tstvls[0])
for xx in range(xlen):
    for xy in range(nlen):
        xkntst[xx] += x_tstvls[xx][xy][0]
    xkntst[xx] /= len(x_tstvls[xx])

# COMMAND ----------

yknact = closscaled[trndatasize:]
yknact = invrs_func(yknact)

# COMMAND ----------

k = 25
ykntst = []
for xx in xkntst:
    ykntst.append(knFunc(kntrn, xx, k))


# COMMAND ----------

knpreds = []
ylen = len(ykntst)
for xx in range(ylen):
    knpreds.append([ykntst[xx]])
    
knpreds = invrs_func(knpreds)
 
tmpVal1 = (yknact - knpreds)
tmpVal2 = tmpVal1**2
tmpVal3 = np.mean(tmpVal2)
rtmnsqr_ktst = rtmnsqr_func(yknact, knpreds)
print("RootMeanSquareErr on prediction :", rtmnsqr_ktst)

tmp2 = abs(tmpVal1)
mae_ktst = np.mean(tmp2)
print("MeanAbsoluteErr on prediction    :", mae_ktst)
 
mse_ktst = tmpVal3
print("MeanSquareErr on prediction      :", mse_ktst)

# COMMAND ----------

dtvals = datetmp.collect()
closvals = clostmp.collect()
plt.title('Train vs Validation vs Prediction using k-NN')
plt.legend(['Train','Validation','Predictions'])
plt.xlabel('Year')
plt.ylabel('Stock price')
plt.plot(dtvals[:trndatasize], closvals[:trndatasize])
plt.plot(dtvals[trndatasize:],closvals[trndatasize:])
plt.plot(dtvals[trndatasize:],knpreds)
