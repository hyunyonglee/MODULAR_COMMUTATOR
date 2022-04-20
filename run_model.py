# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle
import logging.config

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d;

conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
}
logging.config.dictConfig(conf)

os.environ["OMP_NUM_THREADS"] = "68"

Lx = int(sys.argv[1])
Ly = int(sys.argv[2])
h = float(sys.argv[3])
CHI = int(sys.argv[4])
IS = sys.argv[5]
PATH = sys.argv[6]

K = 1.0
Kx = K
Ky = K
Kz = K

model_params = {
    "Lx": Lx,
    "Ly": Ly,
    "Kx": Kx,
    "Ky": Ky,
    "Kz": Kz,
    "h": h
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.KITAEV(model_params)

product_state = ["up","down"] * int(M.lat.N_sites/2)
# product_state = ["up"] * int(M.lat.N_sites)
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if IS == 'random':
    TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 


dchi = int(CHI/5)
chi_list = {}
for i in range(5):
    chi_list[i*10] = (i+1)*dchi

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.2,
        'disable_after': 50
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-10
    },
    'lanczos_params': {
            'N_min': 5,
            'N_max': 20
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-10,
    'max_S_err': 1.0e-6,
    'max_sweeps': 150
}


eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

mag_x = psi.expectation_value("Sigmax")
mag_y = psi.expectation_value("Sigmay")
mag_z = psi.expectation_value("Sigmaz")
EE = psi.entanglement_entropy()

# Measure Flux
Fluxes = []
for i in range(0,Lx):
    for j in range(0, Ly):

        if (j==(Ly-1)): 
            dj = 2*Ly
        else:
            dj = 0

        i0 = 2*Ly*i + 2*j + 1
        i1 = i0 + 1 - dj
        i2 = i0 + 2 - dj

        i3 = i0 + 2*Ly+1 - dj
        i4 = i0 + 2*Ly
        i5 = i0 + 2*Ly-1

        flux = psi.expectation_value_term([('Sigmax',i0),('Sigmay',i1),('Sigmaz',i2),('Sigmax',i3),('Sigmay',i4),('Sigmaz',i5)])
        Fluxes.append(flux)

ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

file_Energy = open(PATH+"/Energy.txt","a")
file_Energy.write(repr(h) + " " + repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")
file_EE = open(PATH+"/Entanglement_Entropy.txt","a")
file_EE.write(repr(h) + " " + "  ".join(map(str, EE)) + " " + "\n")
file_Ws = open(PATH+"/Flux.txt","a")
file_Ws.write(repr(h) + " " + "  ".join(map(str, Fluxes)) + " " + "\n")
file_Sx = open(PATH+"/Sx.txt","a")
file_Sx.write(repr(h) + " " + "  ".join(map(str, mag_x)) + " " + "\n")
file_Sy = open(PATH+"/Sy.txt","a")
file_Sy.write(repr(h) + " " + "  ".join(map(str, mag_y)) + " " + "\n")
file_Sz = open(PATH+"/Sz.txt","a")
file_Sz.write(repr(h) + " " + "  ".join(map(str, mag_z)) + " " + "\n")

file_STAT = open( (PATH+"logs/Stat_h_%.2f.txt" % h) ,"a")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

filename = PATH + 'mps/psi_h_%.2f.pkl' % h
with open( filename, 'wb') as f:
    pickle.dump(psi, f)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
