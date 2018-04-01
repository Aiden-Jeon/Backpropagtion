# Backpropagtion
Backpropagation with numpy

# this is example code to use my model
import pandas as pd
import numpy as np

import layers
import optimizer

data = pd.read_csv('mnist_train_100.csv')

y_data = data.values[:,0]
x_data = data.values[:,1:]
y_data = pd.get_dummies(y_data).values

W1 =layers.weight([784,40])
W2 = layers.weight([40,20])
W3 = layers.weight([20,10])
x_t =layers.train_data(x_data.shape)

layer1 = layers.layer(x_t,W1,np.zeros(40))
layer2 = layers.layer(layer1,W2,np.zeros(20))
layer3 = layers.layer(layer2,W3,np.zeros(10))
layer = [layer1,layer2,layer3]

optimizer.GradientDescent(x_data,y_data,layer,lr=0.1,loss_type='mse',epoch=10000)

'''
0.06060606060606061
0.3838383838383838
0.6565656565656566
0.8080808080808081
0.8383838383838383
0.8787878787878788
0.9090909090909091
0.9292929292929293
0.9393939393939394
0.9494949494949495
'''
