import cigvis
from functions import * 
from cigvis import colormap


# Inputs.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
meander_dir = './Distmap/Meandering/size256_dx25m'  # Meandering channel distance map directory.
tribu_dir = './Topography/Tributary/size256_dx25m'  # Distributary channel distance map database directory.
output_dir = './output'  # Model output directory.
# Meandering channel parameters.
meander_params = [dict(n_channel=2, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=[0.1, 0.3], 
                       incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                       epsilon_range=[0, 0], database=None)]
# Tributary channel parameters.
tribu_params = [dict(n_channel=2, z_range=[0.5, 0.8], hfill_range=[20, 20], 
                     vpfill_std_range=[0, 0], epsilon_range=[0.1, 0.5])]
mute = False
seed_val = 14  # Random seed value.

# Meandering channel database.
meander_database = {}
for item in os.listdir(meander_dir):
    if item.startswith('.'):
        continue
    meander_database[item] = os.listdir(os.path.join(meander_dir, item))
meander_database['dir'] = meander_dir

# Distributary channel database.
tribu_database = {}
tribu_database['id'] = os.listdir(tribu_dir)
tribu_database['dir'] = tribu_dir

# Initialize the model.
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

# Add meandering channels.
model.add_meandering_channel_from_database(params=meander_params, 
                                            database=meander_database, 
                                            seed=seed_val,
                                            instance_label=False, 
                                            replacement=True, 
                                            mute=mute)

# Add tributary channel networks.
model.add_tributary_channel_from_database(params=tribu_params, 
                                          database=tribu_database,
                                          instance_label=False,
                                          replacement=True,   
                                          seed=seed_val)

# Add subamrine canyon.
model.add_submarine_channel(N=1, 
                            X_pos_range=[0.2, 1.0], 
                            Y_pos_range=[0.2, 0.8], 
                            Z_pos_range=[0.3, 0.4], 
                            strike_range=[0, 360], 
                            W_range=[300, 400], 
                            D_range=[30, 40], 
                            kl_range=[50, 60], 
                            Cf_range=[0.07, 0.08], 
                            co_offset=20, 
                            h_oxbow_range=[5, 5],  
                            h_levee_range=[0.5, 0.5], 
                            w_levee_range=[6000, 8000],
                            vp_pointbar_range=[2500, 2700],
                            vp_oxbow_range=[2700, 2900], 
                            vp_levee_range=[2700, 2900], 
                            vp_cap_std=300, 
                            n_iter_range=[1000, 1500], 
                            dt=0.1, 
                            save_iter=10, 
                            t_inci=None, 
                            t_aggr=None, 
                            n_inci_range=[4, 4], 
                            n_aggr_range=[4, 4],
                            kvi_range=[10, 10], 
                            kva_range=None, 
                            dt_vertical=0.05, 
                            rt_vertical=0.5,  
                            delta_s=None, 
                            n_bends_range=[40, 50], 
                            perturb_range=[0.01, 0.03], 
                            pad_up=1, 
                            pad_down=0,
                            instance_label=False,  
                            seed=seed_val, 
                            mute=mute)

# Smooth Vp model.
model.smooth(param=['vp', 'channel'], 
                sigma=1.5, 
                mute=mute)

# Add dipping.
model.add_dipping(a_range=[0.01, 0.04], b_range=[0.01, 0.04], seed=seed_val, mute=mute)

# Add folds.
model.add_fold(N=10, sigma_range=[0.12, 0.15], A_range=[0.02, 0.04], d_fold=3000, zscale=1.0,  
                sync=True, seed=seed_val, mute=mute)

# Add faults.
model.add_fault(N=5, 
                theta_range=[60, 90],
                d_max_range=[0.04, 0.06],  
                d_fault=2000, 
                seed=seed_val, 
                mute=mute)

# Resampling.
model.resample_z(param=['vp', 'channel', 'facies'], 
                 z_range=(100, 1380), 
                 mute=mute)

# Compute P-wave impedance.
model.compute_Ip(rho=2.4)

# Compute reflection coefficients.
model.compute_rc(mute=mute)

# Make synthetic seismic data.
freq = random.uniform(40, 60)
model.make_synseis(mark_channel=True, 
                    plot_wavelet=False, 
                    f_ricker=freq, 
                    wavelet_type='ricker', 
                    length=0.1, 
                    mute=mute)

# 
os.makedirs(output_dir, exist_ok=True)

# Save data.
model.Ip.tofile(output_dir + '/ip.dat')
model.seismic.tofile(output_dir + '/seismic.dat')
model.seis_label.tofile(output_dir + '/label.dat')
model.facies.tofile(output_dir + '/facies.dat')

# Visualization.
fip = output_dir + '/ip.dat'
fsx = output_dir + '/seismic.dat'
flx = output_dir + '/label.dat'
ffx = output_dir + '/facies.dat'
ni, nx, nt = 256, 256, 256
ip = np.fromfile(fip, dtype=np.float32).reshape((ni, nx, nt))
sx = np.fromfile(fsx, dtype=np.float32).reshape((ni, nx, nt))
lx = np.fromfile(flx, dtype=np.uint8).reshape((ni, nx, nt))
fx = np.fromfile(ffx, dtype=np.uint8).reshape((ni, nx, nt))
cmap_lx = colormap.set_alpha_except_min('jet', alpha=0.5)
colors = ['lightgray', 'green', 'gold', '#c09a6b', 'saddlebrown']
values = np.unique(fx)
cmap_fx = colormap.custom_disc_cmap(values, colors)
# Sedimentary facies model.
node0 = cigvis.create_slices(fx, cmap=cmap_fx)
node0 += cigvis.create_colorbar_from_nodes(node0, 'Facies', select='slices')
node0 += cigvis.create_axis(fx.shape, 'axis', axis_pos='auto', 
                            intervals=[25, 25, 5], 
                            axis_labels=['X (m)', 'Y (m)', 'Z (m)'], 
                            ticks_font_size=40, 
                            labels_font_size=40)
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
# Display.
cigvis.plot3D(nodes=[node0, node1, node2, node3], 
              grid=[2, 2], 
              share=True, 
              size=(2100, 1600), 
              cbar_region_ratio=0.125)
