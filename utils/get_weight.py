from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from autograd import grad
import autograd.numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel

def get_weight(Xs,ys,Xt,yt_hat):
    ms, mt = len(Xs), len(Xt)
    X = np.vstack((Xs, Xt))
    y = np.hstack((ys, yt_hat))

    epsilon = 0.001
    pairwise_dist = pairwise_distances(X, X, 'sqeuclidean')
    sigma = np.median(pairwise_dist[pairwise_dist!=0])

    K = rbf_kernel(X, X, gamma=1.0 / sigma) * np.array(y[:,None]==y, dtype=np.float64)   # kernel matrix
    Ks, Kt = K[:ms], K[ms:]

    def obj(theta):
        likelihood = np.mean(np.dot(Kt, theta)) - np.log(np.mean(np.exp(np.dot(Ks, theta))))
        return -likelihood + epsilon * np.dot(theta, theta)

    theta0 = np.zeros(ms+mt)
    result = minimize(obj, theta0, jac=grad(obj), method='L-BFGS-B', options={ 'disp': False}) #, 'maxiter':10
    theta = result.x

    w = np.exp(np.dot(Ks, theta)) / np.mean(np.exp(np.dot(Ks, theta)))
    return w
