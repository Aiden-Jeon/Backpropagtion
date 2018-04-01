import numpy as np
import activation
def weight(shape=[None,None]):
    return np.random.normal(size = shape) / np.sqrt(shape[0]/2)

def layer(X,W,b,active='sigmoid'):
    if type(X) is dict:
        X_t = X['out']
        out = activation.forward(X_t.dot(W)+b,active)
        out_back = activation.backward(out,active)
        return dict(out=out, out_back=out_back, weight=W, bias=b, active=active, net= X_t)
    else:
        X_t = np.array(X)
        out = np.array(X_t).dot(W)+b
        out = activation.forward(out,active)
        out_back = activation.backward(out,active)
        return dict(out=out, out_back=out_back, weight=W, bias=b, active=active, net= X_t)

def train_data(shape=[None,None]):
    return np.zeros(shape[1])
