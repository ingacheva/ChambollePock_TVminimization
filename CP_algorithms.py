from __future__ import division
import numpy as np
from image_operators import *

# ----
VERBOSE = 1
# ----

def filter_projections(proj_set, filt=False):
    if filt == True:
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real
    else:
        return proj_set

def power_method(A, data, v, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    A : projection matrix
    data : acquired sinogram
    v : matrix
    '''
    x = (A.T * filter_projections(data, filt=False)).reshape(v.shape)
    n_angles, l_x = data.shape
    con = 1.0 #np.pi/2.0/n_angles
    for k in range(0, n_it):
        x = (A.T * filter_projections((A*x).reshape(data.shape) * con, filt=False)).reshape(v.shape) - div(gradient(x))
        s = np.sqrt(norm2sq(x))
        x /= s
    return np.sqrt(s)

def chambolle_pock(A, data, v, Lambda, L, n_it, return_energy=True):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)

    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    '''

    sigma = 1.0/L
    tau = 1.0/L
    n_angles, l_x = data.shape
    x = 0*v
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0
    con = 1.0 #np.pi/2.0/n_angles

    if return_energy: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma * ((A*x_tilde).reshape(data.shape) * con - data))/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau * (div(p) - (A.T * filter_projections(q, filt=False)).reshape(v.shape))
        #x[x < 0] = 0
        x_tilde = x + theta * (x - x_old)
        # Calculate norms
        if return_energy:
            fidelity = 0.5 * norm2sq((A * x).reshape(data.shape) * con - data)
            tv = norm1(gradient(x))
            energy = 1.0 * fidelity + Lambda * tv
            en[k] = energy
            if (VERBOSE and k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_energy: return en, x
    else: return x

def calculate_l2(A, sino, rec):
    l2 = tv = norm1(gradient(x))
    return l2


def conjugate_gradient_TV(A, sino, v, Lambda, mu=1e-4, n_it=300):
    '''
    Conjugate Gradient algorithm to minimize the objective function
        0.5*||A*x - d||_2^2 + Lambda*TV_mu (x)

    A : projection matrix
    sino : acquired data as a sinogram
    v : image
    Lambda : parameter weighting the TV regularization
    mu : parameter of Moreau-Yosida approximation of TV (small positive value)
    n_it : number of iterations
    '''

    n_angles, l_x = sino.shape
    con = np.pi/2.0/n_angles
    x = 0*v # start from 0
    grad_f = -(A.T * filter_projections(sino)).reshape(v.shape)
    grad_F = grad_f + Lambda*grad_tv_smoothed(x, mu)
    d = -np.copy(grad_F)
    en = np.zeros(n_it)
    for k in range(0, n_it):
        grad_f_old = grad_f
        grad_F_old = grad_F
        ATAd = (A.T * filter_projections((A*d).reshape(sino.shape) * con)).reshape(v.shape)
        # Calculate step size
        alpha = mydot(d, -grad_F_old)/mydot(d, ATAd)
        # Update variables
        x = x + alpha*d
        grad_f = grad_f_old + alpha*ATAd
        grad_F = grad_f + Lambda*grad_tv_smoothed(x,mu)
        beta = mydot(grad_F, grad_F - grad_F_old)/norm2sq(grad_F_old) # Polak-Ribiere
        if beta < 0:
            beta = 0
        d = -grad_F + beta*d
        # Energy
        fid = norm2sq((A*x).reshape(sino.shape) * con-sino)
        tv = tv_smoothed(x,mu)
        eng = fid + Lambda * tv
        en[k] = eng
        if VERBOSE and (k % 10 == 0):
            print("%d : Energy = %e \t Fid = %e\t TV = %e" %(k, eng, fid, tv))
        # Stoping criterion
        if np.abs(alpha) < 1e-34: # TODO : try other bounds
            print("Warning : minimum step reached, interrupting at iteration %d" %k)
            break;
    return en, x
