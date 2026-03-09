import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    #note that lr is learning rate
    #backprop : xk=x-lr*grad
    #yp=Xw+b
    n_samples,n_features=X.shape
    w=np.zeros(n_features)
    b=0.0
    #steps means how many times we have to run the function
    for _ in range(steps):
        z=X@w + b
        
        p=_sigmoid(z)
        
        grad_w=(X.T @ (p-y))/n_samples

        grad_b=np.sum(p-y)/n_samples
        w=w-lr*grad_w
        b=b-lr*grad_b
        
    return (w,b)
        
    pass