#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


def generateSyntheticData(n, d):
    
    mean = [0]*d
    variance = np.eye(d)
    inputData = []
    outputData = []
    w = np.random.multivariate_normal(mean,np.eye(d))
    
    for i in range(n):
        x = np.random.multivariate_normal(mean,100*variance)
        e = np.random.multivariate_normal([0],np.eye(1))
        y = np.dot(w,x)+e
        #print (y)
        #print (x)
        inputData.append(x)
        outputData.append(y)
        
    return inputData, outputData, w


# In[3]:


def findMax(a):
    n = len(a)
    return max(a)

def findMin(a):
    n = len(a)
    return min(a)


# In[4]:


def normalizeData(inputData, outputData, n, d):
    
    for i in range(n):
        a = inputData[i]
        m1 = findMax(a)
        m2 = findMin(a)
        for j in range(d):
            a[j] = (a[j]-m2)/(m1-m2)
        inputData[i] = a
    
    m3 = findMax(outputData)
    m4 = findMin(outputData)
    for i in range(n):
        outputData[i] = (outputData[i]-m4)/(m3-m4)
    
    return inputData, outputData


# In[5]:


def laplaceFunction(s1, s2, e):
    d = s1.shape[0]
    #print(d)
    z1 = np.random.laplace(size=(d,d),scale = 1/e)
    z2 = np.random.laplace(size=(d,1),scale = 1/e)
    return s1+z1,s2+z2


# In[6]:




r =[]
for n in range(100,1000):
    d = 10
    if (n%10 == 0):
        print ("===============================================")
        print (n)

    inputData, outputData, w = generateSyntheticData(n,d)
    w = np.matrix(w)

    norm_indata, norm_outdata = normalizeData(inputData, outputData, n, d)
    data_in = (norm_indata)
    data_out = (norm_outdata)

    di = [[],[],[],[],[]]
    dits =[[],[],[],[],[]]
    do = [[],[],[],[],[]]
    dots = [[],[],[],[],[]]
    
    di[0] = data_in[:int(4*n/5)]
    #print (len(di[0]))
    #print (len(di[0][1]))
    dits[0] = data_in[int(4*n/5):]
    #print (len(di[0]))

    do[0] = data_out[:int(4*n/5)]
    dots[0] = data_out[int(4*n/5):]
    #print (len(do[0]))


    di[1] = data_in[int(n/5):]
    dits[1] = data_in[:int(n/5)]


    do[1] = data_out[int(n/5):]
    dots[1] = data_out[:int(n/5)]
    #print (len(do[1]))

    di[2] = data_in[int(2*n/5):]
    di[2].extend(data_in[:int(n/5)])
    dits[2] = data_in[int(1*n/5):int(2*n/5)]


    do[2] = data_out[int(2*n/5):]
    do[2].extend(data_out[:int(n/5)])
    dots[2] = data_out[int(1*n/5):int(2*n/5)]
    #print (len(do[2]))
    #print ("*************************************")


    di[3] = data_in[int(3*n/5):]
    di[3].extend(data_in[:int(2*n/5)])
    dits[3] = data_in[int(2*n/5):int(3*n/5)]


    do[3] = data_out[int(3*n/5):]
    do[3].extend(data_out[:int(2*n/5)])
    dots[3] = data_out[int(2*n/5):int(3*n/5)]
    #print (len(do[3]))


    di[4] = data_in[int(4*n/5):]
    di[4].extend(data_in[:int(3*n/5)])
    dits[4] = data_in[int(3*n/5):int(4*n/5)]


    do[4] = data_out[int(4*n/5):]
    do[4].extend(data_out[:int(3*n/5)])
    dots[4] = data_out[int(3*n/5):int(4*n/5)]
    #print (len(do[4]))



    rr = []
    terror = []
    for e in [0,1,2,3,4]:
        di[e] = np.matrix(di[e])
        #print (len(di[e]))
        #print (len(di[e][0]))
        #print ("*************************************")
        do[e] = np.matrix(do[e])
        dits[e] = np.matrix(dits[e])
        dots[e] = np.matrix(dots[e])
        s1 = np.dot(np.transpose(di[e]),di[e])
        s2 = np.dot(np.transpose(di[e]),do[e])
        s1_1 = s1
        s2_1 = s2
        #s1_bar , s2_bar = laplaceFunction(s1_1,s2_1,0.1)
        try:
            w_bar = np.matmul(np.linalg.inv(s1_1),s2_1)
        except:
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print ("!!!!!!!!!!!!!!EXCEPTION!!!!!!!!!!!!!!!!!!!!!")
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
        y = np.matmul(np.transpose(w_bar), np.transpose(dits[e]))
        er = np.sqrt(np.sum(np.square(dots[e]-y))/10)
        #rms_error = np.sqrt(np.sum(np.square(w_bar-w))/10)
        #print (rms_error)
        #r.append(rms_error)
        rr.append(er)
        terror.append(e)
        #print (r)
    r.append(sum(rr)/len(rr))
plt.plot([i for i in range(100,1000)],r)    
#plt.plot([0.1,0.2,0.5,1,2,5,10],r)
plt.show()

