import cigvis
from functions import *
from cigvis import colormap 


# Inputs.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
output_dir = './output'
seed_val = 12  # Random seed value.
mute = False  # True: print less info. False: print all info.

# Initialize the geomodel.
model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                 resolution=[dx, dy, dz], 
                 mute=mute)

# Create horizontal layerd Vp model.
random.seed(seed_val)
vp_noise = random.uniform(300, 400)
model.add_vp(h_layer_list=[(50, 100), (50, 200), (200, 300)],  # Layer thickness ranges of three formations (shallow, middle, and deep formations).
             fm_list=[0.3, 0.7],  # Bottom boundaries of the shallow and middle formations, as a fraction of the model's depth.
             vp_list=[(2500, 3500), (3500, 5000), (5000, 6500)],  # Layer's Vp range in each formation (m/s).
             vp_diff_list=[(300, 500), (200, 400), (100, 300)],  # Vp's difference range between two consecutive layers in each formation.
             vp_disturb=vp_noise,  # Vp perturbations in each layer.
             smooth=False,
             seed=seed_val, 
             mute=mute)

# Add channels.
model.add_meandering_channel(N=1, 
                             X_pos_range=[0.2, 0.3],  # Range of channel centerline's X-coordinate which the model starts at (fraction of the total channel length). 
                             Y_pos_range=[0.4, 0.6],  # Channel Y-coordinate range (fraction of geomodel's Y-range).
                             Z_pos_range=[0.6, 0.8],  # Channel Z-coordinate range (fraction of geomodel's Z-range).
                             strike_range=[0, 360],  # Channel strike direction range (degree, from North). 
                             W_range=[200, 250],  # Channel width range used for centerline simulation (m).
                             D_range=[15, 20],  # Channel depth range used for centerline simulation (m).
                             kl_range=[40, 50],  # Migration rate constant range for centerline simulation (m/year).
                             Cf_range=[0.06, 0.08],  # Chezy's friction coefficient range for centerline simulation. 
                             n_iter_range=[1000, 1500],  # Number of simulation range for centerline simulation. 
                             kv=0,  # Vertical incision rate of the channel trajectory (m/year).
                             aggr=0,  # Vertical aggradation rate of the channel trajectory (m/year).
                             t_incision=None,  # Vertical incision time period.
                             t_aggradation=None,  # Vertical aggradation time period.
                             Wx_range=[300, 300],  # Channel width used to define channel geometry (m).
                             Dx_range=[15, 15],  # Channel depth used to define channel geometry (m).
                             epsilon_range = [0, 0.5],  # Vp contrast between channel fill and its upper layer.
                             incision_mode='random',  # Channel cross-section shape. 1: U-shaped, 2: V-shaped, 'random': U- or V-shaped.
                             seed=seed_val, 
                             instance_label=False,  # Whether to make instance label.
                             mute=mute)

# Smooth Vp and channel model.
model.smooth(param=['vp', 'channel'], sigma=1.5)

# Add dipping.
model.add_dipping(a_range=[0.01, 0.04], b_range=[0.01, 0.04], seed=seed_val, mute=mute)
    
# Add folds.
model.add_fold(N=10, sigma_range=[0.08, 0.10], A_range=[0.05, 0.10], d_fold=500, zscale=1.5,  
               sync=False, seed=seed_val, mute=mute)

# Add faults.
model.add_fault(N=5, 
                theta_range=[60, 90],
                d_max_range=[0.04, 0.06],  
                d_fault=2000, 
                seed=seed_val, 
                mute=mute)

# Resample the geomodel.
model.resample_z(param=['vp', 'channel'],  # Vp model and channel model.
                 z_range=(100, 1380),  # Crop models between z_range. 
                 mute=mute)

# Compute P-wave impedance.
model.compute_Ip(rho=2.4)

# Seismic.
model.compute_rc(mute=mute)
model.make_synseis(f_ricker=25, 
                   mark_channel=True, 
                   plot_wavelet=False)

# Create output directory.
os.makedirs(output_dir, exist_ok=True)

# Save data.
model.Ip.tofile(output_dir + '/ip.dat')  # Impedance model.
model.seismic.tofile(output_dir + '/seismic.dat')  # Seismic data.
model.seis_label.tofile(output_dir + '/label.dat')  # Channel labels.

# Visualization.
fip = output_dir + '/ip.dat'
fsx = output_dir + '/seismic.dat'
flx = output_dir + '/label.dat'
ni, nx, nt = 256, 256, 256
ip = np.fromfile(fip, dtype=np.float32).reshape((ni, nx, nt))
sx = np.fromfile(fsx, dtype=np.float32).reshape((ni, nx, nt))
lx = np.fromfile(flx, dtype=np.uint8).reshape((ni, nx, nt))
cmap_lx = colormap.set_alpha_except_min('jet', alpha=0.5)
# Impedance model.
node1 = cigvis.create_slices(ip, cmap='jet')
node1 += cigvis.create_colorbar_from_nodes(node1, 'Impedance', select='slices')
node1 += cigvis.create_axis(ip.shape, 'axis', axis_pos='auto', 
                            intervals=[25, 25, 5], 
                            axis_labels=['X (m)', 'Y (m)', 'Z (m)'], 
                            ticks_font_size=40, 
                            labels_font_size=40)
# Seismic volume.
node2 = cigvis.create_slices(sx, cmap='Petrel', clim=[-0.3, 0.3])
node2 += cigvis.create_colorbar_from_nodes(node2, 'Seismic', select='slices')
node2 += cigvis.create_axis(sx.shape, 'axis', axis_pos='auto', 
                            intervals=[25, 25, 5], 
                            axis_labels=['X (m)', 'Y (m)', 'Z (m)'], 
                            ticks_font_size=40, 
                            labels_font_size=40)
# Channel masks on top of seismic volume.
node3 = cigvis.create_slices(sx, cmap='gray', clim=[-0.3, 0.3])
node3 = cigvis.add_mask(node3, lx, cmaps=cmap_lx, interpolation='nearest')
node3 += cigvis.create_colorbar_from_nodes(node3, 'Seismic', select='slices')
node3 += cigvis.create_axis(sx.shape, 'axis', axis_pos='auto', 
                            intervals=[25, 25, 5], 
                            axis_labels=['X (m)', 'Y (m)', 'Z (m)'], 
                            ticks_font_size=40, 
                            labels_font_size=40)
# Display
cigvis.plot3D(nodes=[node1, node2, node3], 
              grid=[1, 3], 
              share=True, 
              size=(2100, 800), 
              cbar_region_ratio=0.125)
