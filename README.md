mfGPR
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/Ferg-Lab/mfGPR/workflows/CI/badge.svg)](https://github.com/Ferg-Lab/mfGPR/actions?query=workflow%3ACI)
<!--  [![codecov](https://codecov.io/gh/Ferg-Lab/mfGPR/branch/main/graph/badge.svg)](https://codecov.io/gh/Ferg-Lab/mfGPR/branch/main) -->


Multi-fidelity Gaussian process regression

Seemlessly perform multi-fidelity (and multi-objective) Gaussian process regression. This code is largely a wrapper for [GPy](https://github.com/SheffieldML/GPy) implementing the multi-fidelity Gaussian process regression described in *Perdikaris P., Raissi M., Damianou A., Lawrence N. D. and Karniadakis G. E. 2017Nonlinear information fusion algorithms for data-efficient multi-fidelity modellingProc. R. Soc. A.473* ([paper](https://doi.org/10.1098/rspa.2016.0751), [code](https://github.com/paraklas/NARGP)), with some added functionality for heteroskedastic noise and multi-objective model structuring.


Getting Started
===============


Installation
------------
An environment file `env.yaml` is provided that contains the base dependencies. With [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed:

```
$ conda env create --file mfGPR
$ source activate mfGPR
```

With the environment active the package can then be installed:

```
$ git clone https://github.com/Ferg-Lab/mfGPR.git
$ cd ./mfGPR
$ pip install .
```

Usage
-------
Example Jupyter notebooks demonstrating the functionality outlined here are available in the `examples` directory.

Gaussian process regression models can be defined and trained by passing a dictionary containing the data intended to train the models. For example, for a simple model trained on some (`X`, `Y`) data pairs:

```python
from mfGPR.mfGPR import mfGPR

# define data
data = {'model': {'data':[X, Y]}}

# train model
models = mfGPR(data=data)

# predict on some test points
mean, std = models['model'].predict(X_test)
```

More complex multi-fidelity models can then be defined by specifying a conditioning models treated as the low-fidelity source model:

```python
from mfGPR.mfGPR import mfGPR

# define data
data = {'low': {'data':[X_low, Y_low]},
        'mid': {'data': [X_mid, Y_mid], 'condition': 'mid'},
        'high': {'data': [X_high, Y_high], 'condition': 'mid'},
        }

# train model
models = mfGPR(data=data)

# predict on some test points
mean_low, std_low = models['low'].predict(X_test)
mean_mid, std_mid = models['mid'].predict(X_test)
mean_high, std_high = models['high'].predict(X_test)

# print graphical structure of the mfGPR
models
```

This structure corresponds to a three-level GPR with `mid`-level model conditioned on posterior of the GPR trained on the `low` data, and the `high`-level then conditioned on the posterior of the `mid`-level model. The `_repr_html_` for `models` in a Jupyter notebook will generate a graphical representation of this model structure for simpler visualization of the hierarchical GPR structre:


Another functionality is a multi-objective structure where data from multible objects can be used to condition a higher fidelity model:

```python
from mfGPR.mfGPR import mfGPR

# define data
data = {'low_0': {'data':[X_low_0, Y_low_0]},
        'low_1': {'data': [X_low_1, Y_low_1]},
        'mid': {'data': [X_mid, Y_mid], 'condition': ['low_0', 'low_1']},
        #'mid': {'data': [X_mid, Y_mid], 'condition': ['low_0', 'low_1'], 'theta':[0.5, 0.5]}, # with scalarization explicitly specified 
        'high': {'data': [X_high, Y_high], 'condition': 'mid'},
        }

# train model
models = mfGPR(data=data, n_splits=5, cv_discretization=11)

# predict on some test points
mean_low_0, std_low_0 = models['low_0'].predict(X_test)
mean_low_1, std_low_1 = models['low_1'].predict(X_test)
mean_mid, std_mid = models['mid'].predict(X_test)
mean_high, std_high = models['high'].predict(X_test)

# print graphical structure of the mfGPR
models
```
Operationally, this multi-objective definition performs a scalarization of the the low-fidelity functions and feeds the scalarized posterior as the low-fidelity posterior to the higher fidelity model conditioning on these two low-fidelity models. The scalarization can either be defined by passing a predefined scalarization as a `theta` key in the data dictionary, or if no `theta` key is provided cross-validation is performed to determine the optimal values of `theta` by performing `n_splits`-fold cross validation over a grid of possible `theta` values discretized according to `cv_discretization`.   

Lastly, cross-validation can be peformed after the models have been fit by passing the `'cv':True` in the data dictionary: 

```python
from mfGPR.mfGPR import mfGPR

# define data
data = {'low': {'data':[X_low, Y_low]},
        'mid': {'data': [X_mid, Y_mid], 'condition': 'mid'},
        'high': {'data': [X_high, Y_high], 'condition': 'mid', 'cv':True},
        'high_vanilla': {'data': [X_high, Y_high], 'cv':True},
        }

# train model
models = mfGPR(data=data)

# predict on some test points
mean_low, std_low = models['low'].predict(X_test)
mean_mid, std_mid = models['mid'].predict(X_test)
mean_high, std_high = models['high'].predict(X_test)
mean_high_vanilla, std_high_vanilla = models['high_vanilla'].predict(X_test)

# get cross validation MSE
high_cv_MSE = models['high']['cv_MSE']
high_vanilla_cv_MSE = models['high_vanilla']['cv_MSE']

# print graphical structure of the mfGPR
models
```

References
-------

### Copyright

Copyright (c) 2023, Kirill Shmilovich


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
