# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pynlo
from scipy.constants import c
from scipy.fftpack import next_fast_len
from tqdm import tqdm
import time
import tables
import gc

fft = np.fft.fft
fftshift = np.fft.fftshift


# %% --------------------------------------------------------------------------
# def fft_divide(x, N_ft, fsc=1.0):
#     n = x.size
#     assert (n / N_ft).is_integer(), "must be able to divide array into equal arrays!"
#     if next_fast_len(int(n / N_ft)) != int(n / N_ft):
#         print("not using fastest fft length in computation!")

#     print("initializing arrays!")
#     X = np.zeros((N_ft**2, n // N_ft), dtype=np.complex128)
#     p = 2 * np.pi * np.arange(n) / n
#     exp_p = np.zeros(n, dtype=np.complex128)
#     ft_i = np.zeros(n // N_ft, dtype=np.complex128)
#     print(f"dividing into {N_ft} fft computations, utilizing array of size {N_ft, n}")

#     print("calculating fft's")
#     for i in tqdm(range(N_ft)):
#         ft_i[:] = fft(a_t[i::N_ft])
#         X[N_ft * i : N_ft * (i + 1)] = ft_i[:]
#     X.resize((N_ft, n))

#     print("applying phase factors!")
#     for i in tqdm(range(1, N_ft)):
#         exp_p[:] = np.exp(-1j * p[:] * i)
#         X[i] *= exp_p[:]

#     print("calculating sum!")
#     xf = np.zeros(n, dtype=np.complex128)
#     for i in tqdm(range(N_ft)):
#         xf[:] += X[i]
#     print("final fftshift and multiplying by fsc")
#     xf[:] = fftshift(xf[:]) * fsc
#     return xf


def fft_divide_memmap(x, N_ft, fsc=1.0):
    n = x.size
    assert (n / N_ft).is_integer(), "must be able to divide array into equal arrays!"
    if next_fast_len(int(n / N_ft)) != int(n / N_ft):
        print("not using fastest fft length in computation!")

    file = tables.open_file("_overwrite.h5", "w")
    atom = tables.ComplexAtom(16)
    array = file.create_earray(
        file.root,
        "re",
        atom=atom,
        shape=(0, n),
    )

    print("initializing arrays!")
    p = 2 * np.pi * np.arange(n) / n
    exp_p = np.zeros(n, dtype=np.complex128)
    ft_i = np.zeros(n // N_ft, dtype=np.complex128)
    print(f"dividing into {N_ft} fft computations, utilizing array of size {N_ft, n}")

    print("calculating fft's")
    for i in tqdm(range(N_ft)):
        X = np.zeros((N_ft, n // N_ft), dtype=np.complex128)
        ft_i[:] = fft(a_t[i::N_ft])
        X[:] = ft_i[:]

        shape = X.shape
        X.resize(X.size)
        array.append(X[np.newaxis, :])
        X.resize(shape)

        del X
        gc.collect()

    print("applying phase factors!")
    for i in tqdm(range(1, N_ft)):
        exp_p[:] = np.exp(-1j * p[:] * i)
        array[i] *= exp_p[:]

    print("calculating sum!")
    xf = np.zeros(n, dtype=np.complex128)
    for i in tqdm(range(N_ft)):
        xf[:] += array[i]
    print("final fftshift and multiplying by fsc")
    xf[:] = fftshift(xf[:]) * fsc

    file.close()
    return xf


# %% --------------------------------------------------------------------------
n = 2**10
v_min = c / 2e-6
v_max = c / 1e-6
v0 = c / 1550e-9
e_p = 1e-9
t_fwhm = 50e-15
min_time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(n, v_min, v_max, v0, e_p, t_fwhm, min_time_window)

# %% --------------------------------------------------------------------------
n_pulse = 2000000
a_t = fftshift(pulse.a_t)
a_t = np.tile(a_t, n_pulse)

t1 = time.time()
a_v = fft_divide_memmap(a_t, 5, pulse.dt)
# a_v = fftshift(fft(a_t)) * pulse.dt
t2 = time.time()

print(f"finished in {t2 - t1} seconds")

fig, ax = plt.subplots(1, 1)
plt.plot(abs(a_v) ** 2)
