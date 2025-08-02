from functions import * 


def parallel(i: int, 
             xmin: float, xmax: float, 
             ymin: float, ymax: float, 
             zmin: float, zmax: float, zcut: float,
             dx: float, dy: float, dz: float, 
             channel_params: dict, channel_database: dict, 
             replacement: bool = False,
             instance_label: bool = False,   
             mute: bool = True, seed_val: int = None, 
             output_dir: str = './'):
    
    print(f'Model{i} begin')
    
    # Initialize the model.
    model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                     resolution=[dx, dy, dz], 
                     mute=mute)

    # Create Vp model.
    random.seed(seed_val)
    vp_noise = random.uniform(300, 400)
    model.add_vp(h_layer_list=[(70, 150), (70, 150), (70, 150)], 
                 fm_list=[0.3, 0.7], 
                 vp_list=[(2500, 3500), (3500, 5000), (5000, 6500)],
                 vp_diff_list=[(500, 800), (800, 1000), (800, 1000)],  
                 vp_disturb=vp_noise, 
                 smooth=False,   
                 seed=seed_val, 
                 mute=mute)

    # Add channels.
    model.add_tributary_channel_from_database(params=channel_params, 
                                                 database=channel_database,
                                                 instance_label = instance_label, 
                                                 replacement=replacement,  
                                                 seed=seed_val, 
                                                 mute=mute)
    
    # Smooth Vp model.
    model.smooth(sigma=1.5, 
                 param=['vp', 'channel'], 
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
    
    os.makedirs(output_dir, exist_ok=True)
    
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
database_dir = './Topography/Tributary/size256_dx25m'  # Channel topography database directory.
output_dir = './dataset/tributary'  # Model output directory.
channel_params = [dict(n_channel=3, z_range=[0, 1], hfill_range=[20, 20], 
                       vpfill_std_range=[0, 0], epsilon_range=[0, 0]), 
                  dict(n_channel=3, z_range=[0, 1], hfill_range=[20, 20], 
                       vpfill_std_range=[0, 0], epsilon_range=[0, 0.5]), 
                  dict(n_channel=3, z_range=[0, 1], hfill_range=[20, 20], 
                       vpfill_std_range=[0, 0], epsilon_range=[0.5, 1])]  # Channel parameters.
instance_label = False
replacement=False
mute = True
n_cores = 35  # Number of CPUs used for computing.
if n_cores is None:
    n_cores = multiprocessing.cpu_count() 

# Construct channel database.
channel_database = {}
channel_database['id'] = os.listdir(database_dir)
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