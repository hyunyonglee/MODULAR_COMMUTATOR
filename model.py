# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain
from tenpy.models.lattice import Honeycomb
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['KITAEV']


class KITAEV(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "KITAEV")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 2)
        Kx = model_params.get('Kx', 1.)
        Ky = model_params.get('Ky', 1.)
        Kz = model_params.get('Kz', 1.)
        h = model_params.get('h', 0.)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')

        site = SpinHalfSite(conserve=None)

        order = 'Cstyle'
        # order = ("standard", (True, True, False) , (0, 1, 2))
        lat = Honeycomb(Lx=Lx, Ly=Ly, sites=site, bc=bc, bc_MPS=bc_MPS, order = order)
        
        CouplingModel.__init__(self, lat)


        # magnetic field
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite( -h, u, 'Sigmax')
            self.add_onsite( -h, u, 'Sigmay')
            self.add_onsite( -h, u, 'Sigmaz')

        # x-bond
        u1 = 0 
        u2 = 1 
        dx = [0, 0]
        self.add_coupling( Kx, u1, 'Sigmax', u2, 'Sigmax', dx)
        
        # y-bond
        u1 = 1 
        u2 = 0 
        dx = [1, 0]
        self.add_coupling( Ky, u1, 'Sigmay', u2, 'Sigmay', dx)
        
        # z-bond
        u1 = 1 
        u2 = 0 
        dx = [0, 1]
        self.add_coupling( Kz, u1, 'Sigmaz', u2, 'Sigmaz', dx)


        
        MPOModel.__init__(self, lat, self.calc_H_MPO())

        