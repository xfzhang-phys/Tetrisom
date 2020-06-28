#!/usr/bin/env python
"""
Rebin the original imaginary-time data.
"""
from sys import argv
from numpy import loadtxt, mean, std

print("Please check you arguments: temp nbin")

# Read Gtau from ctau.dat
ctau = loadtxt('ctau.dat')
Ntau = ctau[0].shape[0]
Nsample = ctau.shape[0]

_boltz = 8.617333e-2    # in unit of meV/K
temp = float(argv[1])
beta = 1.0 / (_boltz * temp)
_dt = beta / (Ntau - 1)
_nbin = int(argv[2])
ctau = ctau.T
ctau = ctau.reshape((Ntau, _nbin, int(Nsample/_nbin)))
Nsample = _nbin
ctau = mean(ctau, axis=2)

# Output Gtau.dat and Sigma.dat
gtau_avg = mean(ctau, axis=1)
gtau_std = std(ctau, axis=1)
with open('Gf.dat', 'w') as f:
    for it in range(Ntau):
        tau = it * _dt
        f.write("%20.12lf %24.16lf %20.12lf\n" % (tau, gtau_avg[it], 0.0))

with open('Sigma.dat', 'w') as f:
    for it in range(Ntau):
        tau = it * _dt
        f.write("%20.12lf %24.16lf %20.12lf\n" % (tau, gtau_std[it], 0.0))

