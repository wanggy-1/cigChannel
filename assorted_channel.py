import sys
sys.path.append('..')
from functions import * 


# Inputs.
n_model = 500  # Number of models.
seed_val = None  # Random seed value.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1500, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
meander_database_dir = '/data/CIG/ChannelSeisSim/ChannelDatabase/Distmap/Meandering/size256_dx25m'  # Meandering channel distance map database directory.
distri_database_dir = '/data/CIG/ChannelSeisSim/ChannelDatabase/Distmap/Distributary/size256_dx25m/V1'  # Distributary channel distance map database directory.
output_dir = '/data/CIG/ChannelSeisSim/ChannelSeisDatabase/V1/Assorted'  # Model output directory.
meander_params = [dict(n_channel=3, W_range=[150, 550], D_range=[10, 40], z_range=[0.0, 0.2], 
                       incision_mode='random', vpfill_std_range=[0, 300], hfill_range=[3, 5], 
                       vpfill_range=[3000, 6500], epsilon_range=[0, 1], 
                       database=None)]  # Meandering channel parameters.
distri_params = [dict(n_channel=3, W_range=[200, 500], D_range=[10, 40], z_range=[0.2, 0.4], 
                       incision_mode='random', vpfill_std_range=[0, 300], hfill_range=[3, 5], 
                       vpfill_range=[3000, 6500], epsilon_range=[0, 1])]
n_submarine = 2
mute = True 

# Meandering channel database.
meander_channel_database = {}
for item in os.listdir(meander_database_dir):
    meander_channel_database[item] = os.listdir(os.path.join(meander_database_dir, item))
meander_channel_database['dir'] = meander_database_dir

# Distributary channel database.
distri_channel_database = {}
distri_channel_database['id'] = os.listdir(distri_database_dir)
distri_channel_database['dir'] = distri_database_dir

# Generate seismic data.
for i in range(n_model):
    sys.stdout.write('\rProcessing...%.2f%%[%d/%d]' % ((i+1)/n_model*100, i+1, n_model))
    
    # Initialize the model.
    model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                     resolution=[dx, dy, dz], 
                     mute=mute)

    # Create Vp model.
    random.seed(seed_val)
    vp_noise = random.uniform(300, 500)
    model.add_vp(h_layer_range=[20, 200], 
                 fm_list=[0.2, 0.7], 
                 vp_list=[(3000, 4000), (4000, 5500), (5500, 6500)],
                 vp_diff=300,  
                 vp_disturb=vp_noise, 
                 smooth=False,  
                 seed=seed_val, 
                 mute=mute)

    # Add meandering channels.
    model.add_meandering_channel_from_database(params=meander_params, database=meander_channel_database, seed=seed_val)
    
    # Add distributary channels.
    model.add_distributary_channel_from_database(params=distri_params, database=distri_channel_database, seed=seed_val)
    
    # Make incision plan.
    inci_plan, aggr_plan = channel_vertical_trajectory(inci_range=[0.1, 0.5],
                                              aggr_range=[0.5, 0.9],  
                                              n_incise_range=[2, 3],
                                              n_aggr_range=[2, 3], 
                                              d_inci_range=[0.15, 0.2], 
                                              d_aggr_range=[0.15, 0.2], 
                                              seed_val=seed_val)

    # Add channel.
    model.add_submarine_channel(N=n_submarine, X_pos_range=None, Y_pos_range=None, Z_pos_range=[0.4, 0.8], strike_range=[0, 360], 
                                W_range=[150, 200], D_range=[20, 25], 
                                kl_range=[40, 50], Cf_range=[0.09, 0.11], co_offset=15, 
                                h_oxbow_range=[2, 5], h_channel_range=[2, 5], h_levee_range=[0.1, 0.2], w_levee_range=[3000, 5000], h_cap_range=[5, 8], 
                                vp_oxbow_range=[5000, 6500], vp_pointbar_range=[5000, 6500], vp_levee_range=[5000, 6500], vp_cap_range=[5000, 6500], 
                                n_iter_range=[500, 1500], dt=0.1, save_iter=20, 
                                t_incision=inci_plan, t_aggradation=aggr_plan, kv_range=[6, 7], aggr_range=[5, 6], 
                                delta_s=None, n_bends_range=[20, 30], perturb_range=[0.05, 0.08], 
                                pad_up=1, pad_down=0, seed=seed_val, mute=mute)
    
    # Smooth Vp model.
    model.smooth(sigma=1.5)
    
    # Add dipping.
    model.add_dipping(a_range=[-0.010, 0.010], b_range=[-0.010, 0.010], seed=seed_val, mute=mute)
    
    # Add folds.
    model.add_fold(N=30, sigma_range=[0.06, 0.08], A_range=[0.04, 0.06], sync=False, seed=seed_val, mute=mute)
    
    # Resampling.
    model.resample_z(param=['vp', 'channel'], z_range=(0, 1280), mute=mute)
    
    # Compute P-wave impedance.
    model.compute_Ip(rho=2.4)
    
    # Compute reflection coefficients.
    model.compute_rc(mute=mute)
    
    # Make synthetic seismic data.
    freq = random.uniform(25, 45)
    model.make_synseis(label='channel', plot_wavelet=False, f_ricker=freq, wavelet_type='ricker', length=0.1, mute=mute)
    
    # Save data.
    model.Ip.tofile(output_dir+f'/Ip/{i}.dat')
    model.seismic.tofile(output_dir+f'/Seismic/{i}.dat')
    model.seis_label.tofile(output_dir+f'/Label/{i}.dat')

sys.stdout.write('\n')
print('Done!')
