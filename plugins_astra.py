import astra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CP_algorithms import *

class CP_Plugin(astra.plugin.base):
    """
    ASTRA plugin class implements the Chambolle-Pock algorithm.

    Options:
    'its_PM': number of iteration for Powder Method
    'Lambda': regularation parametr
    """
    astra_name = "CP"

    def initialize(self, cfg, its_PM, Lambda):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']
        self.its_PM = its_PM
        self.Lambda = Lambda

    def run(self, its):
        v = astra.data2d.get_shared(self.vid)
        g = astra.data2d.get_shared(self.sid)
        A = self.W

        re = False
        Lambda = self.Lambda
        print 'Lambda:', Lambda
        #L = 3.01805045502
        L = power_method(A, g, v, n_it=self.its_PM)
        print '||K|| =', L
        if re == True:
            en, u = chambolle_pock(A, g, v, Lambda, L, its, return_energy=re)
            plt.figure()
            plt.plot(en)
            plt.savefig('energy.png')
        else:
            u = chambolle_pock(A, g, v, Lambda, L,  its, return_energy=re)

        v[:] = u.reshape(v.shape)

class CG_Plugin(astra.plugin.base):
    """
    ASTRA plugin class implements a conjugate gradient algorithm.

    Options:
    'its_PM': number of iteration for Powder Method
    """
    astra_name = "CG"

    def initialize(self, cfg):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']

    def run(self, its):
        v = astra.data2d.get_shared(self.vid)
        g = astra.data2d.get_shared(self.sid)
        A = self.W

        Lambda = 0.64 # weight of TV regularization
        mu = 1e-10 # parameter of TV smoothing
        en, u = conjugate_gradient_TV(A, g, v, Lambda, mu, its)
        plt.figure()
        plt.plot(en)
        plt.savefig('CG_energy.png')
        v[:] = u.reshape(v.shape)
