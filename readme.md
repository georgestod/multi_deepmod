# Discovering PDEs from Multiple Experiments

#### In this repo, we share the code, data and results of the paper (https://arxiv.org/abs/2109.11939)
###
### (1) To promote grouped sparsity:
#### we propose and implement a randomized adaptive group Lasso with stability selection and error control
#### see /pdeX/sparsity_estimators.py
###
### (2) Deep learning based model discovery:
#### we implement the latter sparsity estimator in DeepMod (that we extend to handle multiple experiments)
#### we leverage JAX to perform backward and forward autodiffs
#### see /pdeX/DeepModx.py
### (3) We share the code to reproduce the numerical experiments: 
#### varying parameters (paramsXX.ipynb), varying initial conditions (ICs_XX.ipynb) and different chaotic regimes (chaos_XX.ipynb)
#### where XX = {GL: randomized adaptive Group Lasso (grouped sparsity) or IL: randomized adaptive Lasso (individual sparsity)}
####
####
#### Requirements: conda and pip requirements are shared (see .txt files)
