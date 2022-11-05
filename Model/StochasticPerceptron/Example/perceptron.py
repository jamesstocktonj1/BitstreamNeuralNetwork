import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

N = 100
np.random.seed(1)
X1 = np.vstack([np.random.randn(N//2, 2)/3-.5,np.random.randn(N//2, 2)/3+.5 ])
y1 = np.hstack([np.ones(N//2)*-1, np.ones(N//2)]).reshape(-1, 1)
plt.plot(X1[:N//2,0], X1[:N//2,1], 'rx') 
plt.plot(X1[N//2:,0], X1[N//2:,1], 'bo')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig("./images/class_2d.png", bbox_inches="tight")

w = [0, -1, 1]

def plot_decision(w):
    """assumes w has len 3"""
    x1 = np.linspace(-1.5,1.5, 50)
    w0 = np.ones(50)*w[0] # bias value
    x2 = (x1*w[1] + w0)/-w[2]
    
    plt.plot(x1,x2)
    return 

def plot_class(X, w, i, label, n=None):
    plt.figure()
    plt.plot(X[:N//2,0], X[:N//2,1], 'gx') 
    plt.plot(X[N//2:,0], X[N//2:,1], 'b.')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plot_decision(w)
    if n:
        plt.plot(X[n,0], X[n,1], 'ro')
    plt.axis([-1.2, 1.5, -1.5, 1.5])
    plt.savefig(f"./images/perceptron_{label}_{i}.png", bbox_inches="tight")
    plt.close()
    return
label='ls'
plot_class(X1, w, label, 0)

def perceptron(X,y, w, lr, label):
    #w = [0, 0.5, 1] # initial guess
    XX = np.hstack([np.ones(X.shape[0]).reshape(-1,1),X])# add bias
    correct = -np.ones(y.shape) # set to all incorrectly classified
    i = 0
    while np.sum(correct)<len(y): # assumes linearly separable
        rxy = np.random.choice(np.where(correct[0]<1)[0]) # randomly chosen wrongly assigned point
        plot_class(X,w,i, label, rxy) 
        w = w+lr*y[rxy]*XX[rxy]
        i+=1
        
        # classify
        y_hat = np.dot(XX,w)
        y_hat = (y_hat >0)*2-1 # makes  1,-1 array
        correct = y_hat*y.T>0 # gives 1 for correctly assigned values and -1 otherwise
        
        plot_class(X,w,i, label)
        i+=1
        if i>50:
            break
    return w
        
#perceptron(X1, y1, 0.2, 'ls')

N = 100
np.random.seed(0)
X2 = np.vstack([np.random.randn(N//2, 2)/10+.25,np.random.randn(N//2, 2)/10+.75 ])
y2 = np.hstack([np.ones(N//2)*-1, np.ones(N//2)]).reshape(-1, 1)
plt.plot(X2[:N//2,0], X2[:N//2,1], 'rx') 
plt.plot(X2[N//2:,0], X2[N//2:,1], 'bo')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig("./images/class_2d_nls.png", bbox_inches="tight")


w = [0, 0.5, 1]
for i in range(10):
    w = perceptron(X2, y2, w, 0.2, "epoch_{}".format(i))
    print(w)