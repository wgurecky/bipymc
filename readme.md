About
======

Bayesian Inference for PYthon using Markov Chain Monte Carlo (BiPyMc).

BiPyMc contains short example implementations of common Markov chain Monte Carlo routines.  This package also contains Bayesian optimization routines for 1) finding optimal parameter values when the objective function is too costly for MCMC to be feasible and 2) generating initial parameter guesses for MCMC if the objective function is inexpensive.

This package is intended for educational use only.

Try [emcee](https://arxiv.org/abs/1202.3665),
[pymc3](https://docs.pymc.io/), or Dakota for `real` MCMC applications, however, one can
use BiPyMc as a starting point for implementing their own fancy MCMC samplers.

Implemented Example MCMC Methods
---------------------------------

- Metropolis-Hastings
- Adaptive Metropolis (AM)
- Differential Evolution Metropolis (DE-MC)
- Delayed Rejection Metropolis
- Delayed Rejection Adaptive Metropolis (DRAM)


References
-----------

Braak, C.J.F.T. Statistics and Computing (2006) 16: 239. [https://doi.org/10.1007/s11222-006-8769-1](https://doi.org/10.1007/s11222-006-8769-1)

rXiv:1710.09486  [https://arxiv.org/pdf/1710.09486.pdf](https://arxiv.org/pdf/1710.09486.pdf)


Quickstart
----------

Install depends:

    pip install corner mpi4py numpy scipy matplotlib

Or if using conda:

    conda install -c astropy corner
    conda install mpi4py numpy scipy matplotlib

Install and run examples:

    git clone https://github.com/wgurecky/bipymc.git
    cd bipymc
    python setup.py develop --user
    python examples/ex_line_fit.py

Optional parallel example:

    mpirun -np 4 python examples/ex_para_fit.py


Depends
-------

- python3.2+
- numpy
- scipy
- mpi4py (optional for parallel DE-MC)
- matplotlib (optional for plotting)
- [corner](https://corner.readthedocs.io/en/latest/)  (optional for plotting)


License
--------

BSD3 Clause:

Copyright © 2018 William Gurecky
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the organization nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY William Gurecky ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL William Gurecky BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

