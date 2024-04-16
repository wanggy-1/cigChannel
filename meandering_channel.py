import cigvis, vispy
from functions import *
from cigvis import colormap 


# Inputs.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1500, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
seed_val = 42  # Random seed value.
mute = False  # True: print less info. False: print all info.

# Initialize the geomodel.
model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                 resolution=[dx, dy, dz], 
                 mute=mute)

# Assign Vp.
model.add_vp(h_layer_range=[60, 120],  # Thickness range of each layer (m). 
             fm_list=[0.3, 0.6],  # Formation boundaries depth (three formations), as a fraction of total depth.
             vp_list=[(3000, 4000), (4000, 5000), (5000, 6000)],  # Vp range in each formation (m/s).
             vp_diff=500,  # Minimum Vp difference between consecutive layers (m/s).
             vp_disturb=300,  # Standard deviation of the Vp fluctuation in each layer (m/s). 
             smooth=False,  # Whether to smooth the Vp model, we will do it later. 
             seed=seed_val,  # Random seed.
             mute=mute)

# Add channels.
model.add_meandering_channel(N=1, 
                             X_pos_range=[0.2, 0.3],  # Range of channel centerline's X-coordinate which the model starts at. 
                             Y_pos_range=[0.2, 0.8],  # Channel Y-coordinate range (fraction of geomodel's Y-range).
                             Z_pos_range=[0.2, 0.8],  # Channel Z-coordinate range (fraction of geomodel's Z-range).
                             strike_range=[0, 360],  # Channel strike direction range (degree, from North). 
                             W_range=[200, 250],  # Channel width range used for centerline simulation (m).
                             D_range=[15, 20],  # Channel depth range used for centerline simulation (m).
                             kl_range=[40, 50],  # Migration rate constant range for centerline simulation (m/year).
                             Cf_range=[0.05, 0.06],  # Chezy's friction coefficient range for centerline simulation. 
                             n_iter_range=[1000, 1500],  # Number of simulation range for centerline simulation. 
                             kv=0,  # Vertical incision rate of the channel trajectory (m/year).
                             aggr=0,  # Vertical aggradation rate of the channel trajectory (m/year).
                             t_incision=None,  # Vertical incision time period.
                             t_aggradation=None,  # Vertical aggradation time period.
                             Wx_range=[300, 300],  # Channel width used to define channel geometry (m).
                             Dx_range=[15, 15],  # Channel depth used to define channel geometry (m).
                             epsilon_range = [0.5, 0.5],  # Vp contrast between channel fill and its upper layer.
                             incision_mode='random',  # Channel cross-section shape. 1: U-shaped, 2: V-shaped, 'random': U- or V-shaped.
                             seed=seed_val, 
                             instance_label=False,  # Whether to make instance label.
                             mute=mute)

# Smooth Vp and channel model.
model.smooth(param=['vp', 'channel'], sigma=1.5)

# Add inclination to the geomodel.
model.add_dipping(a_range=[0.01, 0.04],  # Inclination rate in X direction (The larger the more inclined).
                  b_range=[0.01, 0.04],  # Inclination rate in Y direction.
                  seed=seed_val, 
                  mute=mute)

# Add folds to the geomodel.
model.add_fold(N=10,  # Number of folds. 
               sigma_range=[0.10, 0.15],  # Width of the folds (as a fraction of the horizontal extension of the model).
               A_range=[0.02, 0.04],  # Height of the folds (as a fraction of the vertical extension of the model).
               d_fold=3000,  # Minimum horizontal spacing of the folds (m).
               seed=seed_val, 
               mute=mute)

# Resample the geomodel.
model.resample_z(param=['vp', 'channel'],  # Vp model and channel model.
                 z_range=(200, 1480),  # Crop models between z_range. 
                 mute=mute)

# Compute P-wave impedance.
model.compute_Ip(rho=2.4)

# Seismic.
model.compute_rc(mute=mute)
model.make_synseis(f_ricker=25, 
                   mark_channel=True)

# Save data.
model.Ip.tofile('./ip_meandering.dat')  # Impedance model.
model.seismic.tofile('./seismic_meandering.dat')  # Seismic data.
model.seis_label.tofile('./label_meandering.dat')  # Channel labels.

# Visualization.
fip = './ip_meandering.dat'
fsx = './seismic_meandering.dat'
flx = './label_meandering.dat'
ni, nx, nt = 256, 256, 256
ip = np.fromfile(fip, dtype=np.float32).reshape((ni, nx, nt))
sx = np.fromfile(fsx, dtype=np.float32).reshape((ni, nx, nt))
lx = np.fromfile(flx, dtype=np.uint8).reshape((ni, nx, nt))
cmap_lx = colormap.set_alpha_except_min('jet', alpha=0.5)
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
                                   cbar_type='bg')
nd3.append(cbar3)
cigvis.plot3D(nodes=[nd1, nd2, nd3], 
              grid=[1, 3], 
              share=True, 
              size=(1600, 800), 
              cbar_region_ratio=0.12)
vispy.app.run()
