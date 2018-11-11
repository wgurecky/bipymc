#!/usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
# Copyright (c) 2017, William Gurecky
# All rights reserved.
#
# DESCRIPTION: Bayesian optimization using gaussian process regression and
# thompson sampling.  Parallel optimization of an arbitray N dimensional function
# is possible via mpi4py.
#
# OPTIONS: --
# AUTHOR: William Gurecky
# CONTACT: william.gurecky@gmail.com
#==============================================================================
from mpi4py import MPI
import numpy as np
from gpr import *


class bo_optimizer(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self):
        pass


class proposal_generator(object):
    """!
    @brief Generates proposal(s)
    """
    def __init__(self):
        pass

    def acqui_function(self, *params):
        """!
        @brief Acquisition function abstract method.
            Determines where to select the next sample point.
        """
        pass

    def gen_proposal(self):
        """!
        @brief Returns the next sample point.
        """
        pass


class bo_thompson(object):
    pass
