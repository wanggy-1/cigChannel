import cigvis, vispy
from functions import *
from cigvis import colormap


# Inputs.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
mute = False  # True: print less info. False: print all info.
seed_val = 24  # Random seed value.
    
# Initialize the geomodel.
model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                 resolution=[dx, dy, dz], 
                 mute=mute)

# Create Vp model.
model.add_vp(h_layer_range=[60, 120],  # Thickness range of each layer (m).
             fm_list=[0.3, 0.7],  # Formation boundaries depth (three formations), as a fraction of total depth.
             vp_list=[(4000, 5500), (4000, 5500), (4000, 5500)], # Vp range in each formation (m/s).
             vp_diff=1000,  # Minimum Vp difference between consecutive layers (m/s).
             vp_disturb=400,  # Standard deviation of the Vp fluctuation in each layer (m/s). 
             smooth=False,  # Whether to smooth the Vp model, we will do it later.  
             seed=seed_val,  # Random seed.
             mute=mute)

# Add channel.
model.add_submarine_channel(N=1,  # Number of channels. 
                            X_pos_range=[0.2, 0.3],  # Range of channel centerline's X-coordinate which the model starts at.
                            Y_pos_range=[0.2, 0.8],  # Channel Y-coordinate range (fraction of geomodel's Y-range).
                            Z_pos_range=[0.4, 0.6],  # Channel Z-coordinate range (fraction of geomodel's Z-range).
                            strike_range=[0, 360],  # Channel strike direction range (degree, from North).  
                            W_range=[300, 400],  # Channel width range (m). 
                            D_range=[30, 40],  # Channel depth range (m).
                            kl_range=[50, 60],  # Migration rate constant range (m/year). 
                            Cf_range=[0.07, 0.08],  # Chezy's friction coefficient range.  
                            h_levee_range=[0.5, 0.5],  # Natural levee thickness range for each time of deposition/iteration (m). 
                            w_levee_range=[6000, 8000],  # Natural levee width range (m).
                            vp_pointbar_range=[2500, 3500],  # Point bar/Sand bar Vp range (m/s).
                            vp_oxbow_range=[2500, 3500],  # Oxbow lake fill Vp range (m/s).
                            vp_levee_range=[2500, 3500],  # Natural levee layer Vp range (m/s).
                            n_iter_range=[1000, 1000],  # Iteration amount range.
                            n_inci_range=[4, 4],  # Range of vertical incision event amount.
                            n_aggr_range=[4, 4],  # Range of vertical aggradation event amount.
                            kvi_range = [8, 8],  # Incision rate range (m/year). 
                            kva_range = [8, 8],  # Aggradation rate range (m/year). 
                            dt_vertical=0.05,  # Channel vertical movement unit time (as a fraction of total number of iteration). 
                            instance_label=False,  # Whether to make instance label.
                            seed=seed_val, 
                            mute=mute)

# Smooth Vp and channel model.
model.smooth(param=['vp', 'channel'], sigma=1.5)

# Add dipping.
model.add_dipping(a_range=[0.01, 0.04], 
                  b_range=[0.01, 0.04], 
                  seed=seed_val, 
                  mute=mute)

# Add folds.
model.add_fold(N=10, 
               sigma_range=[0.10, 0.15], 
               A_range=[0.02, 0.04], 
               d_fold=3000,
               seed=seed_val, 
               mute=mute)

# Resampling.
model.resample_z(param=['vp', 'channel', 'facies'], # Vp, channel and facies model.
                 z_range=(200, 1480), 
                 mute=mute)

# Compute P-wave impedance.
model.compute_Ip(rho=2.4)

# Compute reflection coefficients.
model.compute_rc(mute=mute)

# Make synthetic seismic data.
model.make_synseis(mark_channel=True,  
                   f_ricker=45,   
                   mute=mute)

# Save data.
model.Ip.tofile('./ip_submarine.dat')
model.seismic.tofile('./seismic_submarine.dat')
model.seis_label.tofile('./label_submarine.dat')
model.facies.tofile('./facies_submarine.dat')

# Visualization.
fip = './ip_submarine.dat'
fsx = './seismic_submarine.dat'
flx = './label_submarine.dat'
ffx = './facies_submarine.dat'
ni, nx, nt = 256, 256, 256
ip = np.fromfile(fip, dtype=np.float32).reshape((ni, nx, nt))
sx = np.fromfile(fsx, dtype=np.float32).reshape((ni, nx, nt))
lx = np.fromfile(flx, dtype=np.uint8).reshape((ni, nx, nt))
fx = np.fromfile(ffx, dtype=np.uint8).reshape((ni, nx, nt))
cmap_lx = colormap.set_alpha_except_min('jet', alpha=0.5)
colors = ['lightgray', 'gold', '#c09a6b', 'saddlebrown']
values = np.unique(fx)
cmap_fx = colormap.custom_disc_cmap(values, colors)
nd1, cbar1 = cigvis.create_slices(ip, 
                                  cmap='jet', 
                                  return_cbar=True, 
                                  label_str='Impedance')
nd1.append(cbar1)
nd2, cbar2 = cigvis.create_slices(sx, 
                                  cmap='seismic', 
                                  return_cbar=True, 
                                  label_str='Seismic', 
                                  clim=[-0.2, 0.2])
nd2.append(cbar2)
nd3, cbar3 = cigvis.create_overlay(bg_volume=sx,
                                   fg_volume=lx,
                                   bg_clim=[-0.2, 0.2],  
                                   bg_cmap='gray',
                                   fg_cmap=cmap_lx,  
                                   return_cbar=True, 
                                   cbar_type='bg', 
                                   label_str='Seismic')
nd3.append(cbar3)
nd4, cbar4 = cigvis.create_slices(fx, 
                                  cmap=cmap_fx, 
                                  label_str='Facies Code', 
                                  return_cbar=True)
nd4.append(cbar4)
cigvis.plot3D(nodes=[nd1, nd2, nd3, nd4], 
              grid=[2, 2], 
              share=True, 
              size=(1600, 800), 
              cbar_region_ratio=0.06)
vispy.app.run()

