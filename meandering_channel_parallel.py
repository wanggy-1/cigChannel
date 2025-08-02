from functions import * 


def parallel(i: int, 
             xmin: float, xmax: float, 
             ymin: float, ymax: float, 
             zmin: float, zmax: float, zcut: float,
             dx: float, dy: float, dz: float, 
             channel_params: dict, channel_database: dict, 
             replacement: bool = True,
             instance_label: bool = False,   
             mute: bool = True, seed_val: int = None, 
             output_dir: str = './'):
    
    print(f'Model {i} begin.')
    
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

    # Add channels.
    model.add_meandering_channel_from_database(params=channel_params, 
                                               database=channel_database, 
                                               seed=seed_val,
                                               instance_label=instance_label, 
                                               replacement=replacement, 
                                               mute=mute)
    
    # Smooth Vp model.
    model.smooth(param=['vp', 'channel'], sigma=1.5, mute=mute)
    
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
    model.resample_z(param=['vp', 'channel'], 
                     z_range=(200, 200+zcut), 
                     mute=mute)
    
    # Compute P-wave impedance.
    model.compute_Ip(rho=2.4)
    
    # Compute reflection coefficients.
    model.compute_rc(mute=mute)
    
    # Make synthetic seismic data.
    freq = random.uniform(25, 45)
    model.make_synseis(mark_channel=True, 
                       plot_wavelet=False, 
                       f_ricker=freq, 
                       wavelet_type='ricker',  
                       length=0.1, 
                       mute=mute)
    
    os.makedirs(output_dir + '/Ip', exist_ok=True)
    os.makedirs(output_dir + '/Seismic', exist_ok=True)
    os.makedirs(output_dir + '/Label', exist_ok=True)
    
    # Save data.
    model.Ip.tofile(output_dir+f'/Ip/{i}.dat')
    model.seismic.tofile(output_dir+f'/Seismic/{i}.dat')
    model.seis_label.tofile(output_dir+f'/Label/{i}.dat')
    
    print(f'Model {i} done.')


# ----------- main function ------------- #
# Inputs.
n_model = 200  # Number of models.
start_num = 0  # Start number of the models.
seed_val = None  # Random seed value.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
zcut = 1280
database_dir = './Distmap/Meandering/size256_dx25m'  # Channel distance map database directory.
output_dir = './dataset/meandering'  # Output directory.
channel_params = [dict(n_channel=3, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=[0, 1], 
                       incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                       epsilon_range=[0, 0], database=None), 
                  dict(n_channel=3, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=[0, 1], 
                       incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                       epsilon_range=[0.5, 1.0], database=None), 
                  dict(n_channel=3, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=[0, 1], 
                       incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                       epsilon_range=[0, 0.5], database=None)]  # Channel parameters.
instance_label = False
replacement = True  # Whether to choose channel randomly with replacement.
mute = True  # Whether to mute printing verbose info.
n_cores = 70  # Number of CPUs used for computing.
if n_cores is None:
    n_cores = multiprocessing.cpu_count()

# Construct channel database dictionary.
channel_database = {}
for folder in os.listdir(database_dir):
     if folder.startswith('.'):
          continue
     channel_database[folder] = os.listdir(os.path.join(database_dir, folder))         
channel_database['dir'] = database_dir

# Generate seismic data.
Parallel(n_jobs=n_cores)(delayed(parallel)(i, 
                                           xmin=xmin, xmax=xmax, dx=dx, 
                                           ymin=ymin, ymax=ymax, dy=dy,
                                           zmin=zmin, zmax=zmax, dz=dz, zcut=zcut,   
                                           channel_params=channel_params, 
                                           channel_database=channel_database,
                                           instance_label=instance_label,  
                                           replacement=replacement, 
                                           mute=mute, seed_val=seed_val, 
                                           output_dir=output_dir) 
                                           for i in range(start_num, start_num+n_model))
