from functions import *


def parallel(i: int, 
             xmin: float, xmax: float, 
             ymin: float, ymax: float, 
             zmin: float, zmax: float, zcut: float,
             dx: float, dy: float, dz: float,  
             instance_label: bool = False,   
             mute: bool = True, 
             seed_val: int = None, 
             output_dir: str = './'):
    
    print(f'Model {i} begin.')
    
    # Initialize the model.
    model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                     resolution=[dx, dy, dz], 
                     mute=mute)

    # Create Vp model.
    random.seed(seed_val)
    vp_noise = random.uniform(300, 500)
    model.add_vp(h_layer_list=[(60, 120), (60, 120), (60, 120)], 
                 fm_list=[0.3, 0.7], 
                 vp_list=[(2500, 4000), (2500, 4000), (2500, 4000)],
                 vp_diff_list=[(800, 1000), (800, 1000), (800, 1000)],  
                 vp_disturb=vp_noise, 
                 smooth=False,   
                 seed=seed_val, 
                 mute=mute)
    
    # 
    z_meandering = [[0, 0.3], [0.3, 0.6], [0.6, 1.0]]
    z_tributary = [[0.3, 0.6], [0.6, 1.0], [0, 0.3]]
    z_submarine = [[0.6, 0.7], [0.0, 0.1], [0.3, 0.4]]
    rid = random.choice([0, 1, 2])
    
    # 
    meandering_channel_params = [dict(n_channel=2, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=z_meandering[rid], 
                                      incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                                      epsilon_range=[0, 0], database=None), 
                                 dict(n_channel=2, W_range=[200, 500], R_range=[10, 12], D_range=None, z_range=z_meandering[rid], 
                                      incision_mode='random', vpfill_std_range=[0, 0], hfill_range=[20, 20], 
                                      epsilon_range=[0.1, 0.8], database=None)]  # Meandering channel parameters.
    
    # 
    tributary_channel_params = [dict(n_channel=2, z_range=z_tributary[rid], hfill_range=[20, 20], 
                                        vpfill_std_range=[0, 0], epsilon_range=[0, 0]), 
                                   dict(n_channel=2, z_range=z_tributary[rid], hfill_range=[20, 20], 
                                        vpfill_std_range=[0, 0], epsilon_range=[0.1, 0.8])]  # Channel parameters.

    # Construct meandering channel database dictionary.
    meandering_channel_database = {}
    for folder in os.listdir(meandering_database_dir):
        if folder.startswith('.'):
            continue
        meandering_channel_database[folder] = os.listdir(os.path.join(meandering_database_dir, folder))         
    meandering_channel_database['dir'] = meandering_database_dir

    # Construct tributary channel database.
    tributary_channel_database = {}
    tributary_channel_database['id'] = os.listdir(tributary_database_dir)
    tributary_channel_database['dir'] = tributary_database_dir
    
    # Add meandering channels.
    model.add_meandering_channel_from_database(params=meandering_channel_params, 
                                               database=meandering_channel_database, 
                                               seed=seed_val,
                                               instance_label=instance_label, 
                                               replacement=True, 
                                               mute=mute)
    
    # Add tributary channels.
    model.add_tributary_channel_from_database(params=tributary_channel_params, 
                                                 database=tributary_channel_database,
                                                 instance_label=instance_label, 
                                                 replacement=False,  
                                                 seed=seed_val, 
                                                 mute=mute)
    
    # Add submarine channels.
    model.add_submarine_channel(N=1, 
                                X_pos_range=[0.2, 1.0], 
                                Y_pos_range=[0.2, 0.8], 
                                Z_pos_range=z_submarine[rid], 
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
                                n_iter_range=[600, 1200], 
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
                                instance_label=instance_label,  
                                seed=seed_val, 
                                mute=mute)
    
    # Smooth Vp model.
    model.smooth(param=['vp'], 
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
    z_start = random.uniform(0, 220)
    model.resample_z(param=['vp', 'channel'], 
                     z_range=(z_start, z_start+zcut), 
                     mute=mute)
    
    # Compute P-wave impedance.
    model.compute_Ip(rho=2.4)
    
    # Compute reflection coefficients.
    model.compute_rc(mute=mute)
    
    # Make synthetic seismic data.
    freq = random.uniform(30, 60)
    model.make_synseis(mark_channel=True, 
                       plot_wavelet=False, 
                       f_ricker=freq, 
                       wavelet_type='ricker', 
                       length=0.1, 
                       mute=mute)
    
    os.makedirs(output_dir + '/Facies', exist_ok=True)
    os.makedirs(output_dir + '/Ip', exist_ok=True)
    os.makedirs(output_dir + '/Seismic', exist_ok=True)
    os.makedirs(output_dir + '/Label', exist_ok=True)
    
    # Save data.
    model.facies.tofile(output_dir+f'/Facies/{i}.dat')
    model.Ip.tofile(output_dir+f'/Ip/{i}.dat')
    model.seismic.tofile(output_dir+f'/Seismic/{i}.dat')
    model.seis_label.tofile(output_dir+f'/Label/{i}.dat')
    
    print(f'No.{i} done!')
    
    
# ----------- main function ------------- #
# Inputs.
n_model = 200  # Number of models.
start_num = 0  # Start number of the models.
seed_val = None  # Random seed value.
xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
zcut = 1280
meandering_database_dir = './Distmap/Meandering/size256_dx25m'  # Meandering channel distance map database directory.
tributary_database_dir = './Topography/Distributary/size256_dx25m'  # Tributary channel topography database directory.
output_dir = './dataset/assorted'  # Output directory.
instance_label = True
mute = True  # Whether to mute printing verbose info.
n_cores = 35  # Number of CPUs used for computing.
if n_cores is None:
    n_cores = multiprocessing.cpu_count()

# Generate seismic data.
Parallel(n_jobs=n_cores)(delayed(parallel)(i, 
                                           xmin=xmin, xmax=xmax, 
                                           ymin=ymin, ymax=ymax, 
                                           zmin=zmin, zmax=zmax, zcut=zcut,
                                           dx=dx, dy=dy, dz=dz,  
                                           instance_label=instance_label,   
                                           mute=mute, 
                                           seed_val=seed_val, 
                                           output_dir=output_dir) for i in range(start_num, start_num+n_model))