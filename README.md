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
* `spectra_window_function.ipynb`: compute the power spectra of the catalog chosen to compute the window function example
* `cookbook-window_function_estimate.ipynb`: to compute the Qls coefficients for a halo map far away from the observer, considering redshift space distortions

## Option for running on a Mac OS X 12 (Monterey)

On a Mac, use anaconda and anaconda environments:

* 1. Install homebrew and then install anaconda ($ brew install anacoda)
* If you already have both skip to #2
* 2. Create the "conda environment" inside which you will run de codes:
  * $ conda create -n MTPK_env -c conda-forge python=3.9 numpy scipy pip camb matplotlib pandas jupyter ipython h5py gsl dak c-compiler gfortran
* 3. Check the environment
  * $ conda env list
* 4. Activate the environment
  * $ conda activate MTPK_env
* 5. Install Corrfunc, colossus, deepdish and mcfit using pip:
  * $ pip install Corrfunc
  * $ pip install colossus
  * $ pip install deepdish
  * $ pip install mcfit
* 6. Testthe packages, run your codes, etc.
* 7. After you are done, close the environment
  * $ conda deactivate

* **NOTE:** After creating the environment, activate it and test to see if Corrfunc is working properly
(Corrfunc is a package needed to obtain the Q_ell window functions in redshift space.
  * After activating the environment, open a python shell and run the test:
    * $ python3
      * > from Corrfunc.tests import tests
      * > tests()
  *  If you get an error message complaining that you are missing the gsl library, this may be because the Anaconda gsl is not the latest one.
  * _Quick fix:_ find the latest library, e.g., libgsl.27.dylib , somewhere else in your system, and then copy it to your conda environment:
    * > cp [your_dir]/libgsl.27.dylib  /usr/local/anaconda3/envs/MTPK_env/lib/libgsl.27.dylib