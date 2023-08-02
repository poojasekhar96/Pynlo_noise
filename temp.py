#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:51:05 2023

@author: pooja
"""

# hello world

# %% Imports

#import os, copy
import time
import numpy as np
rng = np.random.default_rng()

from scipy import interpolate
from scipy.constants import c, pi, h
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = (10**8)
from matplotlib import pyplot as plt, ticker
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['savefig.dpi'] = 600

import pynlo
from pynlo.utility import fft
from pynlo import utility as ut
# %% Pulse

v_min = c/3500e-9#c/3000e-9 (3500)
v_max = c/403e-9# c/500e-9, latest 403nm
v0 = c/1550e-9#c/1064e-9
e_p = 95e-12#13.5e-12
t_fwhm = 50e-15#120e-15

n_points = 2**16#2**18, 2**20,it was 2**19, 2**16
pulse = pynlo_connor.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
print("Frq Res: {:.3g} GHz".format(pulse.dv * 1e-9))
#plt.figure('Time domain')
#plt.plot(pulse.t_grid,pulse.p_t)
v_grid = pulse.v_grid

#%%
dv = pulse.dv
grid = pynlo_connor.utility.TFGrid(n_points, v_max, dv, v0=c/1550e-9)
