from __future__ import division
import numpy as np

#image operators
def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient

def div(grad):
    '''
    Compute the divergence of a gradient
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res

def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = np.abs(x)
    m = res < mu
    res[m] = x[m]**2/(2*mu) + mu/2
    return res

def tv_smoothed(x, mu):
    '''
    Moreau-Yosida approximation of Total Variation
    see Weiss, Blanc-Feraud, Aubert, "Efficient schemes for total variation minimization under constraints in image processing"
    '''
    g = gradient(x)
    g = np.sqrt(g[0]**2 + g[1]**2)
    return np.sum(psi(g, mu))

def grad_tv_smoothed(x, mu):
    '''
    Gradient of Moreau-Yosida approximation of Total Variation
    '''
    g = gradient(x)
    g_mag = np.sqrt(g[0]**2 + g[1]**2)
    m = g_mag >= mu
    m2 = (m == False) #bool(1-m)
    g[0][m] /= g_mag[m]
    g[1][m] /= g_mag[m]
    g[0][m2] /= mu
    g[1][m2] /= mu
    return -div(g)

def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())

def norm1(mat):
    return np.sum(np.abs(mat))

def mydot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())

def proj_l2(g, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g**2, 0))/Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res
