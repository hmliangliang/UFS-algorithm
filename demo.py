# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\4\2 0002 10:48:48
# File:         demo.py
# Software:     PyCharm
#------------------------------------
import numpy as np
from scipy.io import loadmat
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
import UFS
import time

if __name__ == '__main__':
    start = time.time()
    #设置相关参数
    K=3
    d=5
    data = loadmat('Breastw.mat')
    data = data['Breastw']
    data = np.array(data)
    col = data.shape[1]
    label = data[:,[col-1]]#数据的类标签
    data = data[:,0:col-1]#获取数据部分
    data = minmax_scale(data)
    data, seq = UFS.UFS(data,K,d)
    data = np.real(data)
    num = int(2*data.shape[0]/3)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data[0:num,:],label[0:num,0])
    y = model.predict(data[num:data.shape[0],:])
    end = time.time()
    print('本文算法测试结果的准确率为:',sum(y==label[num:data.shape[0],0])/(data.shape[0]-num))
    print('所选取的特征序号为：', seq)
    print('本文算法测试的时间消耗为:',end - start)
