# Multi-Tracer Package (MTPK)

**MTPK** is a open source project written in `Python` that provides Fourier analysis and algorithms
useful in parameter inference analysis for datasets from N-body simulations up to observational surveys.

## Features

With this package you will be able to compute:
* Power spectrum using different estimators:
  * Feldman-Kaiser-Peacock (FKP) - [arXiv:9304022](https://arxiv.org/abs/astro-ph/9304022v1)
  * Multi-Tracer (MT) - [arXiv:1505.04106](https://arxiv.org/abs/1505.04106)
  * Both
* Compute the cross power spectrum for FKP
* Compute multipoles
    * Monopole
    * Dipole
    * Quadrupole
* Incorporate mask effects
* Correct the alias effect using Jing approximation - [arXiv:0409240](https://arxiv.org/abs/astro-ph/0409240)
* Chose your favorite mass assignment scheme:
    * Nearest Grid Point (NGP)
    * Cloud In Cell (CIC)
    * Triangular Shaped Cloud (TSC)
    * Piecewise Cubic Spline (PCS)
* Use windows to obtain the convolved theoretical power spectrum, instead of masking it while performing parameter inference

## Requisites

The libraries required are:

* `numpy`
* `os`
* `sys`
* `glob`
* `time`
* `scipy`
* `pandas`
* `h5py`
* `camb`
* `dask`
* `colossus`
* `matplotlib`
* `deepdish`
* `mcfit`
* `ctypes`
* `Corrfunc`

Otherwise to python camb you can use camb usual installation. You only need to specify the directory of your installation in the programs `camb_spec.py` and `theory.py` as a string to the variable `camb_dir`.

## Usage

In this package we provide some **cookbooks** with interactive environment containing some **recipes** with complete examples on _how to_:

* `cookbook-ExSHalos_maps.ipynb`: compute the power spectra of 3 tracers from 4 halo mock catalogues from the code ExSHalos
* `cookbook-window_function_estimate.ipynb`: to compute the Q$_{\ell}$s coefficients for a halo map far away from the observer, considering redshift space distortions
