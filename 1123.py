# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:34:40 2023
圆偏振光入射金纳米颗粒的散射后焦面特性研究
都是下半部分辐射
@author: Administrator
"""

from pyGDM2 import structures
import numpy as np
from pyGDM2 import tools
from pyGDM2 import propagators
from pyGDM2 import fields
from pyGDM2 import core
from pyGDM2 import materials
import matplotlib.pyplot as plt
import numba
from pyGDM2 import visu
if __name__ == '__main__':
    for i in range(1):
        print(i)
    @numba.njit(parallel=True, cache=True)
    def _calc_repropagation(P, Escat, G_dyad_list):
        if len(P) != G_dyad_list.shape[2]//3:
            raise Exception("polarization and Greens tensor arrays don't match in size!")
        _P = P.flatten()
        for i_p_r in numba.prange(G_dyad_list.shape[0]):
            Escat[i_p_r] = np.dot(G_dyad_list[i_p_r], _P)
            
    def farfield(sim, field_index, 
                    r_probe=None,
                    r=100000., 
                    tetamin=0, tetamax=np.pi/2., Nteta=10, 
                    phimin=0, phimax=2*np.pi, Nphi=36, 
                    polarizerangle='none', return_value='map', 
                    normalization_E0=False):
        """spatially resolved and polarization-filtered far-field scattering 
        """
        if r_probe is None:
            tetalist = np.ones((int(Nteta), int(Nphi)))*np.linspace(tetamin, tetamax, int(Nteta))[:,None]
            philist = np.ones((int(Nteta), int(Nphi)))*np.linspace(phimin, phimax, int(Nphi), endpoint=False)[None,:]
            xff = (r * np.sin(tetalist) * np.cos(philist)).flatten()
            yff = (r * np.sin(tetalist) * np.sin(philist)).flatten()
            zff = (r * np.cos(tetalist)).flatten()
            _r_probe = np.transpose([xff, yff, zff])
        else:
            _r_probe = r_probe
        
        ## --- spherical integration steps
        dteta = (tetamax-tetamin) / float(Nteta-1)
        dphi = (phimax-phimin) / float(Nphi)
        
        ## --- incident field config
        field_params    = tools.get_field_indices(sim)[field_index]
        wavelength      = field_params['wavelength']
        
        ## --- environment
        sim.struct.setWavelength(wavelength)
        conf_dict = sim.dyads.getConfigDictG(wavelength, sim.struct, sim.efield)
        if sim.dyads.n2_material.__name__ != sim.dyads.n3_material.__name__:
            raise ValueError("`farfield` does not support a cladding layer at the moment. " +
                             "It implements only an asymptotic Green's tensor for an " +
                             "environment with a single interface (substrate).")
        ######计算入射场
        
        ######################################
            
            ## --- electric polarization of each discretization cell via tensorial polarizability
        Eint = sim.E[field_index][1]
        alpha_tensor = sim.dyads.getPolarizabilityTensor(wavelength, sim.struct)
        P = np.matmul(alpha_tensor, Eint[...,None])[...,0]
        ## --- Greens function for each dipole
        G_FF_EE = np.zeros((len(sim.struct.geometry), len(_r_probe), 3, 3), 
                           dtype = sim.efield.dtypec)
        sim.dyads.eval_G(sim.struct.geometry, _r_probe, 
                              sim.dyads.G_EE_ff, wavelength, 
                              conf_dict, G_FF_EE)
        
        ## propagate fields 
        G_FF_EE = np.moveaxis(G_FF_EE, 0,2).reshape((len(_r_probe), 3, -1))  # re-arrange for direct matrix-vector multiplication
        Escat = np.zeros(shape=(len(_r_probe), 3), dtype=sim.efield.dtypec)
        _calc_repropagation(P, Escat, G_FF_EE)
        
        Iscat = np.sum((np.abs(Escat)**2), axis=1)
        
        
        if return_value.lower() == 'map':
            if r_probe is None:
                return 1
            else:
                return 1
        elif return_value.lower() in ['efield', 'fields', 'field']:
            if r_probe is None:
                return tetalist, philist, Escat
            else:
                return Escat
        else:
            d_solid_surf = r**2 * np.sin(tetalist) * dteta * dphi
            if return_value.lower() == 'int_es':
                if normalization_E0:
                    env_dict = sim.dyads.getConfigDictG(wavelength, sim.struct, sim.efield)
                    E0 = sim.efield.field_generator(sim.struct.geometry, 
                                                    env_dict, **field_params)
                    I0_norm = np.sum(np.abs(E0)**2, axis=1).max()
                else:
                    I0_norm = 1
    
                
    for i in range(1):
        print(i)
        step = 8    # nm; a smaller step was used for the paper figure
       
        geometry = structures.sphere(step, R=5, mesh='hex')
        #geometry.T[0] = geometry.T[0]+ 0*np.cos(i*3.1415926*2/8)
        #geometry.T[1] = geometry.T[1]+ 100*np.sin(i*3.1415926*2/8)
        #visu.structure(geometry,projection = 'XY')
        geometry.T[0] = geometry.T[0]- 149
        material = materials.silicon()
        struct = structures.struct(step, geometry, material)
        print(" N dipoles:", len(struct.geometry))
        ## --- environment (--> homogenous, oil)
        #n1 = n2 = n3 = 1.0
        n1 = 1.0
        n2 = n3 = 1.0
        spacing = 10000.
        dyads = propagators.DyadsQuasistatic123(n1, n2, n3, spacing=spacing)
        
        NA = 0.922# focusing numerical aperture
          
        f = 10    # lens focal distance (mm)
        f0 = 1     # filling factor
        w0 = f0*f*NA/n2
        
        #定义光束
        ####left hand rotating light
        polarization_state= (1,1j,0,0)
        kwargs_hg = dict(xSpot=0.0, ySpot=0.0, zSpot=80.0, kSign = -1.0,
                      NA = NA, f = f, w0 = w0,theta=None, polarization_state= polarization_state, returnField='E',phase=80.0)
        # kwargs_hg = dict(xSpot=0.0, ySpot=0.0, zSpot=50.0, kSign = -1.0,
        #               NA = NA,theta = 0, f = f, w0 = w0, returnField='E')
        
        
        ####
        ###################注意调整结构的同时不要忘记调节波长
        wavelengths = [533]  # finer spectra were calcualted for the paper
        
        field_generator_hg = fields.focused_beams.HermiteGauss00
        # kwargs_hg = dict(xSpot=0.0, ySpot=0.0, zSpot=60.0, kSign=-1.0,
        #                   NA=NA,theta = 0)
        efield_hg = fields.efield(field_generator_hg,
                                    wavelengths=wavelengths,
                                    kwargs=kwargs_hg)
        sim_hg = core.simulation(struct=struct, efield=efield_hg, dyads=dyads)
        sim_hg.scatter()
        print("fieldindex '0' :", sim_hg.E[0][0])
        
        Nteta=68; Nphi= 180
        teta, phi, Escat_hg = farfield(
                                        sim_hg, field_index=0, r=100000,   
                                        tetamin=np.pi/2, tetamax=np.pi,
                                        Nteta=Nteta, Nphi=Nphi,return_value='efield')
        tetalist = np.ones((int(Nteta), int(Nphi)))*np.linspace(np.pi/2,np.pi ,int(Nteta))[:,None]
        philist = np.ones((int(Nteta), int(Nphi)))*np.linspace(0, 2*np.pi, int(Nphi), endpoint=False)[None,:]
        
        Escat_hg.T[0] = Escat_hg.T[0] #/ (-np.cos(tetalist.flatten()))**0.5
        Escat_hg.T[1] = Escat_hg.T[1] #/ (-np.cos(tetalist.flatten()))**0.5
        Escat_hg.T[2] = Escat_hg.T[2] #/ (-np.cos(tetalist.flatten()))**0.5
        
        # #简单测试
        # Iscat = np.sum((np.abs(Escat_hg)**2), axis=1)
        # plt.figure(figsize=(5, 5),dpi = 500)
        # plt.title("back focal plane", x=0.5, y=1.14)
        # ax = plt.subplot(polar=True)
        # im = ax.pcolormesh(philist, np.pi-tetalist, Iscat.reshape(teta.shape)/np.max(Iscat),cmap='jet')
        # #im.set_clim(0,7000)
        # cbar = plt.colorbar(im, orientation='vertical')
        ##############################################################################
        
        Er = (Escat_hg.T[0] * np.sin(tetalist.flatten()) * np.cos(philist.flatten()) + 
                    Escat_hg.T[1] * np.sin(tetalist.flatten()) * np.sin(philist.flatten()) + 
                    Escat_hg.T[2] * np.cos(tetalist.flatten()))
        Ep  = ( Escat_hg.T[0] * np.cos(tetalist.flatten()) * np.cos(philist.flatten()) + 
                    Escat_hg.T[1] * np.sin(philist.flatten()) * np.cos(tetalist.flatten()) - 
                    Escat_hg.T[2] * np.sin(tetalist.flatten()) )
        Es = (-1*Escat_hg.T[0] * np.sin(philist.flatten()) + Escat_hg.T[1] * np.cos(philist.flatten()) )
        
        #####旋转S2
        # Ex = (-1*Ep*np.cos(0.5235+philist.flatten()) - Es*np.sin(0.5235+philist.flatten()))
        # Ey = (-1*Ep*np.sin(0.5235+philist.flatten()) + Es*np.cos(0.5235+philist.flatten()))
        
        Ex = (-1*Ep*np.cos(philist.flatten()) - Es*np.sin(philist.flatten()))
        Ey = (-1*Ep*np.sin(philist.flatten()) + Es*np.cos(philist.flatten()))
        
        # I = np.abs(Ex)**2 + np.abs(Ey)**2
        # s2 = 2*np.real(Ex*Ey.conjugate())
        # s22 = s2/I
        # plt.figure(figsize=(5, 5),dpi = 500)
        # plt.title("back focal plane", x=0.5, y=1.14)
        # ax = plt.subplot(polar=True)
        # ax.grid(False)
        # im = ax.pcolormesh(philist, np.pi-tetalist, s22.reshape(teta.shape),cmap='jet', vmin=-1, vmax=1)
        # cbar = plt.colorbar(im, orientation='vertical')
        
        # I = np.abs(Ex)**2 + np.abs(Ey)**2
        # s3 = -2*np.imag(Ex*Ey.conjugate())
        # s33=s3/I
        # plt.figure(figsize=(5, 5),dpi = 500)
        # plt.title("back focal plane", x=0.5, y=1.14)
        # ax = plt.subplot(polar=True)
        # ax.grid(False)
        # im = ax.pcolormesh(philist, np.pi-tetalist, s33.reshape(teta.shape),vmin=-1., vmax=1.,cmap='PiYG')
        # cbar = plt.colorbar(im, orientation='vertical')
        
        
        
        angle = 45
        angle1 = angle*np.pi/180*np.ones((int(Nteta), int(Nphi)))
        Epor = Ex*np.cos(angle1.flatten()) + 1j*Ey*np.sin(angle1.flatten())
        Ip = np.abs(Epor)**2
        plt.figure(figsize=(5, 5),dpi = 500)
        plt.title("back focal plane", x=0.5, y=1.14)
        ax = plt.subplot(polar=True)
        plt.title("%s"%(i*45), x=0.5, y=1.14)
        ax.grid(False)
        im = ax.pcolormesh(philist, np.pi-tetalist,np.log(Ip.reshape(teta.shape)),cmap='jet')
        cbar = plt.colorbar(im, orientation='vertical')
        
        plt.figure(figsize=(5, 5),dpi = 500)
        plt.title("back focal plane", x=0.5, y=1.14)
        ax = plt.subplot(polar=True)
        plt.title('%a'%(10*i-100), x=0.5, y=1.14)
        ax.grid(False)
        im = ax.pcolormesh(philist, np.pi-tetalist,np.angle( Epor.reshape(teta.shape)),cmap='jet')
        cbar = plt.colorbar(im, orientation='vertical')
        
        
        #弱项光强值
        Epor = (Ex + 1j*Ey)/2**0.5
        Ip = np.abs(Epor)**2
        plt.figure(figsize=(5, 5),dpi = 500)
        plt.title("back focal plane", x=0.5, y=1.14)
        ax = plt.subplot(polar=True)
        plt.title("%s"%(149), x=0.5, y=1.14)
        ax.grid(False)
        im = ax.pcolormesh(philist, np.sin(np.pi-tetalist),np.log(Ip.reshape(teta.shape)),cmap='jet')
        cbar = plt.colorbar(im, orientation='vertical')
        
        a = np.log(Ip.reshape(teta.shape))
        
        #弱项相位值
        Epor = (Ex + 1j*Ey)/2**0.5
        Ip = np.angle(Epor)
        plt.figure(figsize=(5, 5),dpi = 500)
        plt.title("back focal plane", x=0.5, y=1.14)
        ax = plt.subplot(polar=True)
        plt.title("%s"%((1)*149), x=0.5, y=1.14)
        ax.grid(False)
        plt.yticks([0,0.5,1])
        im = ax.pcolormesh(philist, np.sin(np.pi-tetalist),Ip.reshape(teta.shape),cmap='jet')
        cbar = plt.colorbar(im, orientation='vertical')
        
        
        field_params = tools.get_field_indices(sim_hg)[0]
        wavelength   = field_params['wavelength']
        sim_hg.struct.setWavelength(wavelength)
        eps_env = sim_hg.dyads.getEnvironmentIndices(wavelength, sim_hg.struct.geometry[:1])[0]  # assume structure is fully in one environment
        n_env = np.sqrt(eps_env)
        ## --- get polarizability at wavelength
        alpha_tensor = sim_hg.dyads.getPolarizabilityTensor(wavelength, sim_hg.struct)
        E = sim_hg.E[0][1]
        P = np.matmul(alpha_tensor, E[...,None])[...,0]
        Px,Py,Pz = P.sum(axis=0)
        print(np.angle(Px,deg=1))
        print(np.angle(Pz,deg=1))
        #(np.angle(Px,deg=0)-np.angle(Pz,deg=0))/np.pi*180
        np.abs(Px)/np.abs(Pz)