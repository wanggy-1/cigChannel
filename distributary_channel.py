import sys
sys.path.append('..')
from functions import * 


# Inputs.
n_model = 1  # Number of models.
seed_val = 42  # Random seed value.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1500, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m)
database_dir = '/Users/wangguangyu/Codes/SynologyDrive/Others/soillib-main/tools/basic_hydrology/topography'  # Channel distance map database directory.
output_dir = '/Users/wangguangyu/Data/CIG/ChannelSeisDatabase'  # Model output directory.
channel_params = [dict(n_channel=1, z_range=[0.4, 0.6], 
                       vpfill_std_range=[0, 0], hfill_range=[3, 5], 
                       vpfill_range=[3000, 6500], epsilon_range=[0, 0])]  # Channel parameters.
mute = False 

# Channel database.
channel_database = {}
channel_database['id'] = os.listdir(database_dir)
for item in channel_database['id']:
    if item.startswith('.'):
        channel_database['id'].remove(item)
channel_database['dir'] = database_dir

for i in range(n_model):
    print('Model[%d/%d]' % (i+1, n_model))
    
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
    
    # Create RGT model.
    model.add_rgt(mute=mute)

    # Add channels.
    model.add_distributary_channel_from_database(params=channel_params, 
                                   database=channel_database, 
                                   seed=seed_val)
    
    # Smooth Vp model.
    model.smooth(sigma=1.5, 
                 param=['vp',  
                        'channel'])
    
    # Add dipping.
    model.add_dipping(a_range=[-0.010, 0.010], b_range=[-0.010, 0.010], seed=seed_val, mute=mute)
    
    # Add folds.
    model.add_fold(N=30, sigma_range=[0.06, 0.08], A_range=[0.04, 0.06], sync=False, seed=seed_val, mute=mute)
    
    # Resampling.
    model.resample_z(param=['vp',
                            'rgt',  
                            'channel'], 
                     z_range=(0, 1280), 
                     mute=mute)
    
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
    model.rgt.tofile(output_dir+f'/RGT/{i}.dat')
    os.makedirs(output_dir+f'/Horizon/{i}')
    for j, h in enumerate(model.horizon):
        if h.channel:
            s = extract_isosurface(model.rgt, 
                                   h.rgt)
            s.tofile(output_dir+f'/Horizon/{i}/{j}.dat')

print('Done!')
