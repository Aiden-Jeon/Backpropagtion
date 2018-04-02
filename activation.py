import numpy as np
def forward(x,method='sigmoid'):
    x = np.array(x)
    if method.lower() == 'sigmoid':
        return 1/(1+np.exp(-x))
    if method.lower() == 'relu':
        x[x<0] = 0 
        return x
    if method.lower() == 'softmax':
        x = np.exp(x - np.max(x))
        return x / np.sum(x)

def backward(x,method='sigmoid'):
    x = np.array(x)
    if method.lower() == 'sigmoid':
        return x*(1-x)
    if method.lower() == 'relu':
        x[x<0] = 0 
        x[x>0] = 1
        return x
    if method.lower() == 'softmax':
        x = np.exp(x - np.max(x))
        return (np.sum(x)-x)/ np.sum(x)
        

