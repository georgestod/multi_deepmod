#############################################################################
# The Adaptive Lasso and Adaptive Group Lasso with stability selection and error control
# Author: Georges Tod 
# Institutions: University of Paris, CRI Research, France
# September 2021
#############################################################################

import numpy as np
import matplotlib.pylab as plt

from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge

from sklearn.utils import shuffle

from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample
from scipy.stats import beta

from mutar import GroupLasso

from joblib import Parallel, delayed
import jax
from jax import random, numpy as jnp


def get_mask_multi(theta,dt,method:str='GL',randomized:bool=True):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    dt = np.array(dt,dtype=np.float64)
    # normalizing dt and library time wise
    normed_dt = dt/np.linalg.norm(dt,axis=1,keepdims=True)
    normed_theta = theta/np.linalg.norm(theta,axis=1,keepdims=True)
    
    # computing mask
    if method == 'GL':
        # internal warm start for acceleration
        clf = GroupLasso(alpha=1e-4, fit_intercept=False,max_iter=1e4,tol=1e-3,warm_start=True) 
        clf.fit(normed_theta, normed_dt[:,:,0])
        coef_ini = clf.coef_shared_, clf.coef_specific_ 
        
        mask,for_plots = Adaptive_Group_Lasso_SS(normed_theta,normed_dt,coef_ini,randomized=randomized)
        maxP_selec = for_plots[0].max(axis=0)
        
    if method == 'IL':
        n_tasks, n_samples, n_features = theta.shape
        masks = []
        for i in range(n_tasks):
            mask,_ = Adaptive_Lasso_SS(normed_theta[i,:,:],normed_dt[i,:,:],randomized=randomized)
            masks.append(mask)
        mask = np.array(masks)
        mask = mask.reshape(n_tasks,n_features,1) # to match coeffs definition
        mask = np.array(mask,dtype=np.uint8)
        maxP_selec = np.array([np.nan])

        
    if method == 'sIL':
        n_tasks, n_samples, n_features = theta.shape
        # stacking all on top of each other
        
        
        
        theta = theta.reshape(n_tasks*n_samples,n_features)
        dt = dt.reshape(n_tasks*n_samples,1)

        
        
        # normalize
        normed_dt = dt/np.linalg.norm(dt,axis=0,keepdims=True)
        normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
        # computing mask
        mask,for_plots = Adaptive_Lasso_SS(normed_theta,normed_dt,randomized=randomized)
        mask = np.array(mask,dtype=np.uint8).reshape(-1,1)
        maxP_selec = for_plots[0].max(axis=0)
        mask = np.repeat(mask.reshape(1,n_features,1),n_tasks,axis=0)

        
    return mask, maxP_selec






def Adaptive_Group_Lasso_SS(X,y,coef_ini,
                     randomized:  bool   = True,
                     alphas:      np.ndarray = None,                  
                     nI:          int = 40, 
                     ratio:       int = 2,
                     n_alphas:    int = 10,
                     n_cores:     int = -1, 
                     eps:         int = 3,
                     d1:          float = 1,
                     d2:          float = 2,
                     seed:        int = 42,
                     super_power: float = 2,               
                     efp:         int = 3,
                     piT:         float = 0.9,
                     tol:         float = 1e-5):
    
    

    n_tasks, n_samples, n_features = X.shape
    y = y[:,:,0]

    n_population  = X.shape[1]
    n_pop_samples = int(n_population/ratio)
    
    
    if alphas is None:
        alpha_max = np.max([np.sqrt(np.sum(((X[i,:,:].T) @ y[i,:]) ** 2)).max()/n_samples for i in range(n_tasks)])
        e_amax = np.log10(alpha_max).round()
        e_amin = e_amax - eps
        if e_amin < -6:
            e_amin = -6
        alphas    = np.logspace(e_amin, e_amax, n_alphas, base=10)

    # randomizing
    np.random.seed(seed=seed)        
    if randomized:
        W = np.random.beta(d1,d2,size=[n_tasks,nI,n_features])
        W[W>1]   = 1
        W[W<0.1] = 0.1
    else:
        W = np.ones([n_tasks,nI,n_features])
            
    def stab_(j, coef_ini):        
        active = np.zeros([n_features,n_alphas])
        for i, alpha in enumerate(alphas):
            idx_ = sample_without_replacement(n_population=n_population,n_samples=n_pop_samples)
            y_train = y[:,idx_]
            X_train = X[:,idx_,:] * np.sqrt(W[:,j,:]).reshape(n_tasks,1,n_features)
            
            ##########################################
            def adaGroupLasso(alpha, coef_ini):
                # initial weights
                coeffRidge = np.array([Ridge(alpha=1e-10,fit_intercept=False).fit(x, y_train[i]).coef_ for i, x in enumerate(X_train)]).T
                weights = (np.linalg.norm(coeffRidge,axis=1,keepdims=True)+ np.finfo(float).eps)**super_power

                X_w = X_train * weights.T[:, None, :]           

                clf = GroupLasso(alpha=alpha, fit_intercept=False,max_iter=1e4,tol=tol,warm_start=True) 
                clf.coef_shared_= coef_ini[0]
                clf.coef_specific_ = coef_ini[1]

                clf.fit(X_w, y_train)
                coef_ini = clf.coef_shared_, clf.coef_specific_ # ship initial guess for future runs
                coef_ = clf.coef_ * weights

                    
                return coef_, coef_ini
            ##########################################

            
            betahat, coef_ini = adaGroupLasso(alpha,coef_ini)
            betahat = betahat * np.sqrt(W[:,j,:]).T

            
            group_norm = np.linalg.norm(betahat,axis=1)
            active[:,i] = (group_norm > 0) * 1/nI
            
        return active

    # selection probabilities
    pSE = Parallel(n_jobs=n_cores)(delayed(stab_)(j, coef_ini) for j in range(nI))
    #pSE = [stab_(j,coef_ini) for j in range(nI)]
    tau = np.array(pSE).sum(axis=0).T
    
    
    # average of selected variables
    q_hat = tau.sum(axis=1)
    # verifying on some upper bound on efp
    ev_region = (q_hat)**2/((2*piT-1)*X.shape[2])    
    idxS = (ev_region<efp).argmax()   
    minLambdaSS = alphas[idxS]
    # selecting variables where the efp is respected
    active_set = (tau[idxS:,:]>piT).any(axis=0)   

    
    mask = np.tile(active_set,[n_tasks,1])
    mask = mask.reshape(n_tasks,n_features,1) # to match coeffs definition
    mask = np.array(mask,dtype=np.uint8)


    for_plots = [tau,piT,pSE,alphas, minLambdaSS, active_set,(ev_region<efp)]
    
    return mask, for_plots







def Adaptive_Lasso_SS(X,y,
                     randomized: bool=True,
                     alphas: np.ndarray = None,                  
                     nI: int = 40, 
                     ratio: int = 2,
                     n_alphas: int = 10,
                     n_cores: int=-1, 
                     eps: int=3,
                     d1: float = 1,
                     d2: float = 2,
                     seed: int = 42,
                     super_power: float = 2,
                     efp:int = 3,
                     piT:float = 0.9,
                     tol: float = 1e-5):
    
    
    
    if alphas is None:
        # computing range of alphas
        alpha_max = np.sqrt(np.sum(((X.T) @ y) ** 2)).max()/X.shape[0]
        e_amax = np.log10(alpha_max).round()
        e_amin = e_amax - eps
        alphas    = np.logspace(e_amin, e_amax, n_alphas, base=10)

    # some params
    n_population = X.shape[0]
    n_samples    = int(n_population/ratio)
    
    # randomizing
    np.random.seed(seed=seed)        
    if randomized:
        W = np.random.beta(d1,d2,size=[nI,X.shape[1]])
        W[W>1]   = 1
        W[W<0.1] = 0.1
        
    else:
        W = np.ones([nI,X.shape[1]])

    def stab_(i, alpha):  
        tau = np.zeros(X.shape[1])
        for j in range(nI):
            idx_ = sample_without_replacement(n_population=n_population,n_samples=n_samples)
            y_train = y[idx_]
            X_train = X[idx_,:] * W[j,:]
            ##########################################
            def adaLasso(alpha):
                n_samples, n_features = X.shape
                weights = (np.abs(Ridge(alpha=1e-10,fit_intercept=False).fit(X, y).coef_).ravel())**super_power
                X_w = X_train * weights[np.newaxis, :]
                clf = Lasso(alpha=alpha, fit_intercept=False,tol=tol)
                clf.fit(X_w, y_train)
                coef_ = clf.coef_ * weights
                return coef_
            ##########################################
            betahat = adaLasso(alpha) * W[j,:]
            active = (np.abs(betahat) > 0) *1/nI
            
            tau = tau + active
            
        return tau

    # selection probabilities
    pSE = Parallel(n_jobs=n_cores)(delayed(stab_)(i, alpha) for i, alpha in enumerate(alphas))
    tau = np.array(pSE)
        
        
    # average of selected variables
    q_hat = tau.sum(axis=1)
    # verifying on some upper bound on efp
    ev_region = (q_hat)**2/((2*piT-1)*X.shape[1])    
    idxS = (ev_region<efp).argmax()   
    minLambdaSS = alphas[idxS]
    # selecting variables where the efp is respected
    active_set = (tau[idxS:,:]>piT).any(axis=0)   
        
    mask = active_set
    mask = np.array(mask,dtype=np.uint8)


    for_plots = [tau,piT,pSE,alphas, minLambdaSS, active_set,(ev_region<efp)]
    
    return mask, for_plots

    
def ir_cond(theta,GT):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    # normalizing
    normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
    
    x1 = normed_theta[:,GT]
    x2 = normed_theta[:,np.setdiff1d(np.arange(0,theta.shape[1]),np.array(GT))]

    metric_PoV = np.abs(np.linalg.inv(x1.T @ x1) @ x1.T @ x2)
    deltaPoV = np.linalg.norm(metric_PoV,1,axis=0).max() 
    
    
    cond_num = np.linalg.cond(normed_theta[:,:])

    return deltaPoV, cond_num

    
def ir_condAL(theta,dt,GT,super_power: float = 2,lambda_ridge: float = 1e-10):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    dt = np.array(dt,dtype=np.float64)

    # normalizing
    normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
    normed_dt = dt/np.linalg.norm(dt,axis=0,keepdims=True)


    weights = (np.abs(Ridge(alpha=lambda_ridge,fit_intercept=False).fit(normed_theta,normed_dt).coef_).ravel())**super_power
    normed_theta_w = normed_theta * weights[np.newaxis, :]

    x1 = normed_theta_w[:,GT]
    x2 = normed_theta_w[:,np.setdiff1d(np.arange(0,theta.shape[1]),np.array(GT))]

    metric_PoV = np.abs(np.linalg.inv(x1.T @ x1) @ x1.T @ x2)
    deltaPoV = np.linalg.norm(metric_PoV,1,axis=0).max() 


    cond_num = np.linalg.cond(normed_theta_w[:,:])

    return deltaPoV, cond_num