#!/usr/bin/env python
"""
Fourier transform from imaginary-time correlation function to Matsubara frequencies correlation.
The code is implemented by cubic spline interpolate instead of discrte Fourier transform directly.
Ref: Phys.Rev.B 84 085128 (2011)
"""
from sys import argv
from numpy import loadtxt, pi, exp, zeros, array, real, mean, std
from scipy.interpolate import  CubicSpline
from scipy.fftpack import ifft
from scipy.integrate import simps

print("Please check you arguments: Temp  [Nom]  [nbin]")

# Read Gtau from Gtau.dat
ctau = loadtxt('ctau.dat')
Ntau = ctau[0].shape[0]
Ngrid = ctau.shape[0]
_N = Ntau - 1

# Get temperature and number of Matsubara frequencies from command line
_boltz = 8.617333e-2    # in unit of meV/K
temp = float(argv[1])
if len(argv) > 1:
    Nom = int(argv[2])
else:
    Nom = 2 * Ntau - 1
if len(argv) > 2:
    _nbin = int(argv[3])
    ctau = ctau.T
    ctau = ctau.reshape((Ntau, _nbin, int(Ngrid/_nbin)))
    Ngrid = _nbin
    ctau = mean(ctau, axis=2).T

_Nom_half = int((Nom - 1) / 2)
beta = 1. / (_boltz * temp)
_dt = beta / _N
_tpdb = 2. * pi / beta
# tau list
tau = [0.]
for i in range(1, Ntau):
    tau.append(i * _dt)
tau = array(tau)

# Calculate Gom
Gom = zeros((Ngrid, Nom))
for ismpl in range(Ngrid):
    # moments for evaluate Gom by cubic spline
    spl = CubicSpline(tau, ctau[ismpl])
    dev1 = spl(tau, nu=1)
    dev2 = spl(tau, nu=2)
    dev3 = spl(tau, nu=3)
    idev3 = ifft(dev3[1:]) * _N
    _M0 = -1.*ctau[ismpl][0] - ctau[ismpl][-1]
    _M1 = dev1[0] - dev1[-1]
    _M2 = -1.*dev2[0] - dev2[-1]
    for iw in range(-1*_Nom_half, 0):
        wn = iw * _tpdb
        _res = _M0 / (1j * wn) + _M1 / ((1j * wn) ** 2) + _M2 / ((1j * wn) ** 3)
        _res += ((1. - exp(1j * wn * _dt)) / (1j * wn) ** 4) * idev3[iw%_N]
        Gom[ismpl][iw+_Nom_half] = real(_res)
    for iw in range(1, _Nom_half+1):
        wn = iw * _tpdb
        _res = _M0 / (1j * wn) + _M1 / ((1j * wn) ** 2) + _M2 / ((1j * wn) ** 3)
        _res += ((1. - exp(1j * wn * _dt)) / (1j * wn) ** 4) * idev3[iw%_N]
        Gom[ismpl][iw+_Nom_half] = real(_res)
    # simpson integration for Gom(0)
    Gom[ismpl][_Nom_half] = simps(ctau[ismpl], tau)

# Output Gom.dat and Sigma.dat
Gom = Gom.T
gom_avg = mean(Gom, axis=1)
gom_std = std(Gom, axis=1)
with open('Gom.dat', 'w') as f:
    for iw in range(Nom):
        om = (iw - _Nom_half) * _tpdb
        f.write("%20.12lf %20.12lf %20.12lf\n" % (om, gom_avg[iw], 0.0))

with open('Sigma.dat', 'w') as f:
    for iw in range(Nom):
        om = (iw - _Nom_half) * _tpdb
        f.write("%20.12lf %20.12lf %20.12lf\n" % (om, gom_std[iw], 0.0))
