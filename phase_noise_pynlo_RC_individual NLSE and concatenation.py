# %% Imports

#import os, copy
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:27:55 2023

@author: fastdaq
"""
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

v_min = c/1800e-9#c/3000e-9 (3500), 1640, 2000
# v_max = c/1206e-9# c/500e-9, latest 403nm, 955
v0 = c/1550e-9#c/1064e-9
e_p = 2.5e-9#95e-12#13.5e-12, 1e-9 for lowerrep. rate, correct 95pJ for 50 fs # 2.5 nJ for 500 fs
t_fwhm = 500e-15#120e-15

n_points = 2**13 #2**18, 2**20,it was 2**19, 2**16
v_max = v_min + n_points * 10e9
pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm, alias = 2)
print("Frq Res: {:.5g} GHz".format(pulse.dv * 1e-9))
v_grid = pulse.v_grid
dv = pulse.dv
# plt.figure('Freq domain')
# plt.semilogy(pulse.v_grid*1e-12,pulse.p_v)
#%%
# defining pulse from transform 
# dir = "C:/Users/Diddams/OneDrive - UCB-O365/Documents/GitHub/Pynlo_noise/"
# osa_data = np.loadtxt(dir + "W0053.csv", delimiter=',', skiprows = 39)# spectrum after ND HNLF + PM1550
# f_osa = c/(osa_data[::-1,0]*1e-9)
# lin_osa = 10**(osa_data[::-1,1]/10)*1e-3/dv # in W 
# plt.figure('osa')
# plt.plot(f_osa,osa_data[:,1])
# p_v = interpolate.InterpolatedUnivariateSpline(f_osa, lin_osa,ext=1)

# pulse = pynlo.light.Pulse.FromPowerSpectrum(p_v, n_points, v_min, v_max)
# pulse.a_v*=np.exp(1j*-8.6e-3*(pulse.v_grid*1e-12)**2 + 1j*-3e-6*(pulse.v_grid*1e-12)**3)# 8.6
# pulse.e_p = 95e-12 # 95e-12
# print(pulse.t_width())

#%% Waveguide from ri_interpolator

import ri_interpolator
thickness = 800e-9#690e-9
width = 2500e-9#750e-9
sim_freqs = ri_interpolator.sim_freqs
sim_oversample = np.linspace(sim_freqs.min(), sim_freqs.max(), sim_freqs.size*100)
sim_n_eff, sim_gamma_mode, sim_aeffs = ri_interpolator.refractive_index_and_gamma(
    [thickness], [width], sim_freqs, mode='Ex')
sim_gamma = sim_gamma_mode/2

gamma_spline = interpolate.InterpolatedUnivariateSpline(
    sim_freqs,
    sim_gamma,
    ext="extrapolate") #1.4632 1/(W km) @ 1064nm

g3_v = pynlo.utility.chi3.gamma_to_g3(pulse.v_grid, gamma_spline(pulse.v_grid))

n_eff_spline = interpolate.InterpolatedUnivariateSpline(
        sim_freqs,
        sim_n_eff,
        ext="extrapolate",
        k=5)

beta_v = pynlo.utility.chi1.n_to_beta(pulse.v_grid, n_eff_spline(pulse.v_grid))

#---- Mode
mode2 = pynlo.medium.Mode(pulse.v_grid, beta_v, g3=g3_v) # media

#%%
# Adding phase noise from PLO
dir = "C:/Users/Diddams/OneDrive - UCB-O365/Documents/GitHub/Pynlo_noise/"  
plo = np.loadtxt(dir + "phase_noise_PLO.txt", delimiter=',',skiprows=0) # L(f) in dBc/Hz
#plo = np.loadtxt(r"/Users/pooja/Documents/Rigol_1GHz.csv", delimiter=',',skiprows=0) # L(f) in dBc/Hz
#plt.semilogx(plo[:,0],plo[:,1])
s_phi_plo = plo[:,1] + 3 # two-sided PSD # dBc/Hz
s_phi_plo_lin = 10**(s_phi_plo/10) # rad^2/Hz
var_phase = np.trapz(s_phi_plo_lin, plo[:,0]) # integrate phase noise density
print(var_phase)

# extending white phase noise till Nyquist freq 5 GHz
f1 = np.linspace(31e6, 5e9, 10000)
f_new = np.concatenate([plo[:,0], f1])
phase_noise_extend = np.ones(len(f1)) * s_phi_plo_lin[-1]
phase_noise_den = np.concatenate([s_phi_plo_lin, phase_noise_extend])
var_phase1 = np.trapz(phase_noise_den, f_new)
print(var_phase1)
'''
RMS phase jitter is sqrt(2.A) where A is the integrated phase noise power
'''
var_phase1_new = 2*var_phase1
print(var_phase1_new)
#%%

# adding amp noise + phase noise for actual case

var = h*pulse.v_grid/(4*pulse.dv)
mean = 0
length = 5e-3
norm = pulse.p_v/np.max(pulse.p_v)
fr= dv
# idx1 = np.argmin(np.abs(pulse.v_grid - (v0 - 0.051e14))) # 0.031
# idx2 = np.argmin(np.abs(pulse.v_grid - (v0 + 0.051e14))) # equally distributed, 7.2 THz BW
CW_LW = 5e3#200e6 # 5 kHz, 1e7 ran
var_pn_CW = 2*pi*CW_LW*pulse.dt # variance of CW laser 


tot_no = 2**3 #2**15 
phase_noise_CW = np.cumsum(np.sqrt(var_pn_CW) * np.random.randn(n_points*tot_no))

inp_p_a_t = np.zeros((tot_no,n_points),dtype=np.complex_)
out_p_a_t = np.zeros((tot_no,n_points),dtype=np.complex_)
for i in range(tot_no):
    # p1 = pynlo.light.Pulse.FromPowerSpectrum(p_v, n_points, v_min, v_max)
    # p1.a_v*=np.exp(1j*(-8.6e-3*(pulse.v_grid*1e-12)**2 + -3e-6*(pulse.v_grid*1e-12)**3))
    # p1.e_p = 95e-12
    p1 = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    # adding intensity noise
    # test = (rng.normal(mean,np.sqrt(var),n_points)) + 1j*rng.normal(mean,np.sqrt(var),n_points)
    # test += (2e-6 * p1.p_v)**0.5 * (rng.standard_normal(n_points) + 1j*rng.standard_normal(n_points))
    # p1.a_v += test # shot noise
    
    phase_noise = rng.normal(0,np.sqrt(var_phase1_new)*1)/(2*np.pi * fr)*1e-1
    p1.a_v*= np.exp(1j *2*pi*(pulse.v_grid-v0)*phase_noise)
    # adding flicker noise
    # f = np.arange(n_points//2 + 1) * dv/n_points
    # white_td = rng.normal(size=n_points)*1 # in rad suppose
    # white_fd = np.fft.rfft(white_td * p1.dt) # rad/Hz^2
    # flicker_fd = white_fd/f**0.5# rad/Hz^2.5
    # flicker_fd[0] = 0 # divide by zero error
    # flicker_td = np.fft.irfft(flicker_fd * n_points * (dv/n_points)) * 1e7# in rad/Hz
    # p1.a_t*= np.exp(1j * flicker_td) # mult v_grid not right, too high
    
    p1.a_t*= np.exp(1j*phase_noise_CW[i*n_points: (i+1)*n_points])     # adding CW phase noise
    inp_p_a_t[i] = p1.a_t
    sim = pynlo.model.NLSE(p1, mode2)
    local_error = 1e-6
    dz = sim.estimate_step_size(local_error=local_error)
    new_pulse, z, a_t, a_v = sim.simulate(length, dz=dz, local_error=local_error, n_records=None, plot=None)
    out_p_a_t[i] = new_pulse.a_t
    if i% (2**4) == 0:
        print(i)
    # print(i)
  
inp_p_a_t.shape = inp_p_a_t.size
out_p_a_t.shape = out_p_a_t.size
# np.save(dir + "tim_pul_train_inp_timjitter_var_phase1_-v0_CWphasenoise" + str(CW_LW) + "_2rt" + str(np.log2(tot_no*n_points)) + ".npy", inp_p_a_t)
# np.save(dir + "tim_pul_train_SCout_timjitter_var_phase1_-v0_CWphasenoise" + str(CW_LW) + "_2rt" + str(np.log2(tot_no*n_points)) + "SiN2500nm.npy", out_p_a_t)
#%%
# SPECTRUM

N_new = tot_no * n_points
del_v = (v_max - v_min)/N_new#len(out_p_a_t)
dt = 1/(N_new * del_v)
inp_train_a_v = fft.fftshift(fft.fft(fft.ifftshift(inp_p_a_t),fsc = dt)) 
out_train_a_v = fft.fftshift(fft.fft(fft.ifftshift(out_p_a_t),fsc = dt)) 

# np.save(dir + "sc_out_timjitter_var_phase1_-v0_CWphasenoise" + str(CW_LW) + "_2rt" + str(np.log2(tot_no*n_points)) + "SiN2500nm.npy", out_p_a_t)
#%%
v_new = np.linspace(v_min, v_max, N_new)
idx1 = np.argmin(np.abs(v_new*1e-12 - c*1e-3/1500)) # c*1e-3/1570
idx2 = np.argmin(np.abs(v_new*1e-12 - c*1e-3/1510))

plt.semilogy(v_new* 1e-12, np.abs(inp_train_a_v)**2, 'o-')
plt.semilogy(v_new* 1e-12, np.abs(out_train_a_v)**2, 'o-')
# plt.plot(v_new[idx2:idx1] * 1e-12, np.abs(out_train_a_v[idx2:idx1])**2, 'o-') 
plt.xlabel('Frequency (THz)')
plt.ylabel('PSD (J/Hz)')

#%%
# trying new fft split

# fft = np.fft.fft
# fftshift = np.fft.fftshift
# ifftshift = np.fft.ifftshift

# out_p_a_t = ifftshift(out_p_a_t)
# # out_p_a_t = ifftshift(out_p_a_t)
# # t1 = time.time()
# fft_divide_mmap(x = out_p_a_t, N_ft = 4, fsc = 1)
# a_v = fftshift(fft(a_t)) * pulse.dt
# t2 = time.time()

# print(f"finished in {t2 - t1} seconds")
# %% ------------------------- load the output fft for viewing / plotting -----

# file = tables.open_file("_overwrite.h5", "r")

# a_v1 = np.array(file.root.data[:])

# file.close()
#%%

# from scipy.optimize import curve_fit
# from scipy.special import wofz
# import pylab
# def gaussian(x, cen1, sigma1):
#     return (np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) # amp1*(1/(sigma1*(np.sqrt(2*np.pi))))
# # lw = 2* np.sqrt(2*np.log(2))*sigma1

# def lor(x, cen1, sigma1):
#     return 1/(((x - cen1)/sigma1)**2 + 1) # amp1*(1/(sigma1*(np.sqrt(2*np.pi))))

# def V(x, cen1, alpha, gamma):
#     """
#     Return the Voigt line shape at x with Lorentzian component HWHM gamma
#     and Gaussian component HWHM alpha.

#     """
#     sigma = alpha / np.sqrt(2 * np.log(2))

#     return np.real(wofz(((x-cen1) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
#                                                            /np.sqrt(2*np.pi)



# id1 = np.argmin(np.abs(v_new*1e-12 - 198.6)) # 191.412441, 201.003116
# test_x = v_new[id1-1500:id1+3000]*1e-12 # id1 - 200, new_pulse1.v_grid
# test_y = np.abs(sc[id1-1500:id1+3000])**2/np.max(np.abs(sc[id1-1500:id1+3000])**2) 
# pk_frq = test_x[np.argmax(test_y)]

# test_y1 = np.abs(sc1[id1-1500:id1+3000])**2/np.max(np.abs(sc1[id1-1500:id1+3000])**2)# new_pulse1.p_v 
# pk_frq1 = test_x[np.argmax(test_y1)]
# popt_gauss, pcov_gauss = scipy.optimize.curve_fit(V, test_x, test_y, bounds = ((pk_frq-1e-3, 1e-6, 70e-6), (pk_frq+1e-3,20e-6, 320e-6)))#, p0=[amp1, cen1, sigma1]), ((pk_frq-1e-3, 80e-6, 10e-6), (pk_frq+1e-3,170e-6, 80e-6)))
# popt_gauss1, pcov_gauss1 = scipy.optimize.curve_fit(V, test_x, test_y1, bounds = ((pk_frq1-1e-3, 0.5e-6,1e-6), (pk_frq1+1.00e-3,2e-6, 6e-6)))# (pk_frq1-1e-4, 1e-6,0.5e-6), (pk_frq1+1.00e-4,6e-6, 3e-6)

# # popt_gauss, pcov_gauss = scipy.optimize.curve_fit(lor, test_x, test_y, bounds = ((pk_frq-1e-5, 100e-6), (pk_frq+1e-5,10e-3)))#, p0=[amp1, cen1, sigma1])
# # popt_gauss1, pcov_gauss1 = scipy.optimize.curve_fit(lor, test_x, test_y1, bounds = ((pk_frq-1e-7, 0), (pk_frq+1.00e-7,18e-6)))
# # popt_gauss, pcov_gauss = scipy.optimize.curve_fit(gaussian, test_x, test_y, bounds = ((pk_frq-1e-4, 0), (pk_frq+1.00e-4,1.5e-6)))#, p0=[amp1, cen1, sigma1])
# # popt_gauss1, pcov_gauss1 = scipy.optimize.curve_fit(gaussian, test_x, test_y1, bounds = ((pk_frq-1e-4, 0), (pk_frq+1e-4,0.5e-6)))#, p0=[amp1, cen1, sigma1])
# perr_gauss = np.sqrt(np.diag(pcov_gauss))

# # voigt linewidth
# fL = 2* popt_gauss1[2]
# fG = 2* popt_gauss1[1]*np.sqrt(2*np.log(2))
# lw_v = 0.5346*fL + np.sqrt(0.2166 * fL**2 + fG**2)
# print(lw_v*1e6)

# test_x_new = np.linspace(test_x.min(), test_x.max(), test_x.size *100)
# plt.plot(test_x, test_y,'o', label = '$10^{7}$ flicker, 0.5 GHz CW linewidth')
# plt.plot(test_x_new, V(test_x_new, *popt_gauss)/np.max(V(test_x_new, *popt_gauss)))
# plt.plot(test_x, test_y1,'o', label = '$10^{5}$ flicker, 10MHz CW linewidth')
# plt.plot(test_x_new, V(test_x_new, *popt_gauss1)/np.max(V(test_x_new, *popt_gauss1)))
# plt.xlabel('Frequency (THz)')
# plt.ylabel('Intensity (a.u.)')
# plt.grid()
# plt.tight_layout()
# plt.legend()
# print(pcov_gauss1)
