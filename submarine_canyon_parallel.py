from functions import *


def parallel(i: int, 
             xmin: float, xmax: float,
             ymin: float, ymax: float, 
             zmin: float, zmax: float, zcut: float, 
             dx: float, dy: float, dz: float,
             n_channel: int, 
             output_dir: str = './', 
             seed_val: int = None, 
             mute: bool = True):
    
    print(f'No.{i} begin...')
    
    # Initialize the model.
    model = GeoModel(extent=[xmin, xmax, ymin, ymax, zmin, zmax], 
                     resolution=[dx, dy, dz], 
                     mute=mute)

    # Create Vp model.
    random.seed(seed_val)
    vp_noise = random.uniform(300, 500)
    model.add_vp(h_layer_list=[(60, 100), (60, 100), (60, 100)],  
                 fm_list=[0.3, 0.7], 
                 vp_list=[(2500, 4000), (2500, 4000), (2500, 4000)],
                 vp_diff_list=[(800, 1000), (800, 1000), (800, 1000)],  
                 vp_disturb=vp_noise, 
                 smooth=False,  
                 seed=seed_val, 
                 mute=mute)

    # Add channel.
    zs = [[0.0, 0.1], [0.4, 0.5], [0.7, 0.8]]
    its = [[600, 800], [800, 1000], [1000, 1200]]
    ids = [0, 1, 2]
    random.shuffle(ids)
    for j in range(n_channel):
        model.add_submarine_channel(N=1, 
                                    X_pos_range=[0.2, 1.0], 
                                    Y_pos_range=[0.2, 0.8], 
                                    Z_pos_range=zs[j], 
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
                                    n_iter_range=its[ids[j]], 
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
    z_start = random.uniform(0, 220)
    model.resample_z(param=['vp', 'channel', 'facies'], 
                     z_range=(z_start, z_start+zcut), 
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data.
    model.Ip.tofile(output_dir+f'/Ip/{i}.dat')
    model.seismic.tofile(output_dir+f'/Seismic/{i}.dat')
    model.seis_label.tofile(output_dir+f'/Label/{i}.dat')
    model.facies.tofile(output_dir+f'/Facies/{i}.dat')
    
    print(f'No.{i} done!') 


if __name__ == '__main__':
    
    n_model = 200  # Number of models.
    start_num = 0  # Start number of the models.
    seed_val = None  # Random seed value.
    xmin, xmax, dx = 0, 6400, 25  # xmin: minimum x-coordinate (m); xmax: maximum x-coordinate (m); dx: cell length (m).
    ymin, ymax, dy = 0, 6400, 25  # ymin: minimum y-coordinate (m); ymax: maximum y-coordinate (m); dy: cell width (m).
    zmin, zmax, dz = 0, 1600, 5  # zmin: minimum z-coordinate (m); zmax: maximum z-coordinate (m); dz: cell height (m).
    zcut = 1280
    n_channel = 3
    output_dir = './dataset/submarine'  # Model output directory.
    mute = True
    n_cores = 70
    
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()
    
    Parallel(n_jobs=n_cores)(delayed(parallel)(i, 
                                               xmin=xmin, xmax=xmax, dx=dx, 
                                               ymin=ymin, ymax=ymax, dy=dy,
                                               zmin=zmin, zmax=zmax, dz=dz,
                                               zcut = zcut, 
                                               n_channel=n_channel,  
                                               output_dir=output_dir, 
                                               seed_val=seed_val, 
                                               mute=mute) for i in range(start_num, start_num+n_model))
