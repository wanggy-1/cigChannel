"""
Functions for generating geologic models and synthetic seismic data.

Created by Guangyu Wang @ USTC
2022.10.01
"""

import random, math, numba, sys, multiprocessing, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage, interpolate
from scipy.spatial import distance
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from prettytable import PrettyTable
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable


class BiharmonicSpline2D:
    """
    2D Bi-harmonic Spline Interpolation.
    """

    def __init__(self, x, y):
        """
        Use coordinates of the known points to calculate the weight "w" in the interpolation function.
        
        :param x: (numpy.ndarray) - x-coordinates of the known points.
        :param y: (numpy.ndarray) - y-coordinates of the known points.
        """
        # x- and y-coordinates of the known points.
        self.x = x
        self.y = y
        
        # Check if the array shape of x and y is identical.
        if not self.x.shape == self.y.shape:
            raise ValueError("The array shape of x- and y-coordinates must be identical. Get x%s and y%s instead." % 
                             (self.x.shape, self.y.shape))
        
        # Flatten the coordinate arrays if they are not.
        if self.x.ndim != 1 and self.y.ndim != 1:
            self.x = self.x.ravel(order='C')
            self.y = self.y.ravel(order='C')
        
        # Calculate the 1D Green function matrix.
        green = np.zeros(shape=[len(self.x), len(self.x)], dtype=np.float32)
        for i in range(len(x)):
            green[i, :] = np.abs(self.x[i] - self.x) ** 3
        
        # Calculate weights.
        if np.linalg.matrix_rank(green) == green.shape[0]:  # The Green matrix is invertible.
            self.w = np.linalg.inv(green) @ self.y
        else:  # The Green matrix is non-invertible.
            self.w = np.linalg.pinv(green) @ self.y  # Use pseudo-inverse.

    
    def __call__(self, x_new):
        """
        Interpolate new points.
        
        :param x_new: (numpy.ndarray) - x-coordinates of the new points.
        
        :return y_new: (numpy.ndarray) - y-coordinates of the new points.
        """
        # Get the array shape of new x-coordinates.
        original_shape = x_new.shape
        
        # Flatten the coordinate arrays if they are not.
        if x_new.ndim != 1:
            x_new = x_new.ravel(order='C')
        
        # Calculate the 1D Green function matrix.
        green = np.zeros(shape=[len(x_new), len(self.x)], dtype=np.float32)
        for i in range(len(x_new)):
            green[i, :] = np.abs(x_new[i] - self.x) ** 3
        
        # Calculate y-coordinates of new points.
        y_new = green @ self.w
        y_new = y_new.reshape(original_shape, order='C')
        
        return y_new


class BiharmonicSpline3D:
    """
    3D Bi-harmonic Spline Interpolation.
    """

    def __init__(self, x, y, z):
        """
        Use coordinates of the known points to calculate the weight "w" in the interpolation function.
        
        :param x: (numpy.ndarray) - x-coordinates of the known points.
        :param y: (numpy.ndarray) - y-coordinates of the known points.
        :param z: (numpy.ndarray) - z-coordinates of the known points.
        """
        # x-, y- and z-coordinates of the known points.
        self.x = x
        self.y = y
        self.z = z
        
        # Check if the array shape of x, y and z is identical.
        if not self.x.shape == self.y.shape == self.z.shape:
            raise ValueError("The array shape of x-, y- and z-coordinates must be identical. Get x%s, y%s and z%s instead." % 
                             (self.x.shape, self.y.shape, self.z.shape))
        
        # Flatten the coordinate arrays if they are not.
        if self.x.ndim != 1 and self.y.ndim != 1 and self.z.ndim != 1:
            self.x = self.x.ravel(order='C')
            self.y = self.y.ravel(order='C')
            self.z = self.z.ravel(order='C')
        
        # Calculate the 2D Green function matrix.
        delta_x = np.zeros(shape=[len(self.x), len(self.x)], dtype=np.float32)
        delta_y = np.zeros(shape=[len(self.y), len(self.y)], dtype=np.float32)
        for i in range(len(x)):
            delta_x[i, :] = self.x[i] - self.x  # Calculate the x-coordinate difference between two points.
            delta_y[i, :] = self.y[i] - self.y  # Calculate the y-coordinate difference between two points.
        mod = np.sqrt(delta_x ** 2 + delta_y ** 2)  # The vector's modulus between two points.
        mod = mod.ravel(order='C')  # Flatten the 2D mod array to 1D.
        green = np.zeros(shape=mod.shape, dtype=np.float32)  # Initialize the Green function matrix.
        
        # Calculate the Green function matrix at non-zero points.
        green[mod != 0] = mod[mod != 0] ** 2 * (np.log(mod[mod != 0]) - 1)
        green = green.reshape(delta_x.shape)  # Reshape the matrix to 2-D array shape.
        
        # Calculate weights.
        if np.linalg.matrix_rank(green) == green.shape[0]:  # The Green matrix is invertible.
            self.w = np.linalg.inv(green) @ self.z
        else:  # The Green matrix is non-invertible.
            self.w = np.linalg.pinv(green) @ self.z  # Use pseudo-inverse.

    
    def __call__(self, x_new, y_new):
        """
        Interpolate new points.
        
        :param x_new: (numpy.ndarray) - x-coordinates of the new points.
        :param y_new: (numpy.ndarray) - y-coordinates of the new points.
        
        :return znew: (numpy.ndarray) - z-coordinates of the new points.
        """
        # Get the array shape of new x-coordinates.
        original_shape = x_new.shape
        
        # Check if the array shape of x and y is identical.
        if not x_new.shape == y_new.shape:
            raise ValueError("The array shape of x- and y-coordinates must be identical. Get x%s and y%s instead." % 
                             (self.x.shape, self.y.shape))
        
        # Flatten the coordinates if they are not.
        if x_new.ndim != 1 and y_new.ndim != 1:
            x_new = x_new.ravel(order='C')
            y_new = y_new.ravel(order='C')
        delta_x = np.zeros(shape=[len(x_new), len(self.x)], dtype=np.float32)
        delta_y = np.zeros(shape=[len(y_new), len(self.y)], dtype=np.float32)
        for i in range(len(x_new)):
            delta_x[i, :] = x_new[i] - self.x
            delta_y[i, :] = y_new[i] - self.y
        mod = np.sqrt(delta_x ** 2 + delta_y ** 2)  # The vector's modulus between two points.
        mod = mod.ravel(order='C')  # Flatten the 2D mod array to 1D.
        green = np.zeros(shape=mod.shape, dtype=np.float32)
        green[mod != 0] = mod[mod != 0] ** 2 * (np.log(mod[mod != 0]) - 1)
        green = green.reshape(delta_x.shape)
        
        # Calculate z-coordinates of new points.
        z_new = green @ self.w
        z_new = z_new.reshape(original_shape, order='C')
        
        return z_new


class GeoModel:
    """
    Geologic model.
    """
    
    def __init__(self, extent, resolution, mute=False):
        """
        Initialize a geologic model with hexahedral cells.
        
        :param extent: (List of floats) - [Xmin, Xmax, Ymin, Ymax, Zmin, Zmax]. Extent of the model (unit: meter).
        :param resolution: (List of floats) - [dX, dY, dZ]. Model's resolution in each dimension (unit: meter).
        :param mute: (Bool) - If True, will not print anything. 
                              Default value is False.
        """
        # Coordinates.
        self.Xmin, self.Xmax = extent[0], extent[1]
        self.Ymin, self.Ymax = extent[2], extent[3]
        self.Zmin, self.Zmax = extent[4], extent[5]
        self.dX, self.dY, self.dZ = resolution[0], resolution[1], resolution[2]
        X = np.arange(start=self.Xmin, stop=self.Xmax, step=self.dX, dtype=np.float32)
        Y = np.arange(start=self.Ymin, stop=self.Ymax, step=self.dY, dtype=np.float32)
        Z = np.arange(start=self.Zmin, stop=self.Zmax, step=self.dZ, dtype=np.float32)
        self.nX, self.nY, self.nZ = len(X), len(Y), len(Z)
        self.X, self.Y, self.Z = np.meshgrid(X, Y, Z)
        
        # Attributes.
        self.vp = None  # P-wave velocity.
        self.Ip = None  # P-wave impedance.
        self.rc = None  # Reflection coefficient.
        self.seismic = None  # Synthetic seismic data.
        self.seis_label = None  # Seismic response marker of channels.
        self.horizon = None  # Horizons.
        self.fault = None  # Fault label (0: non-fault, 1: fault).
        self.facies = None  # Facies label (0: background, 1: channel fill, 2: point-bar, 3: natural levee, 4: oxbow lake).
        self.channel = None  # River channel label (0: non-channel, 1: channel).
        self.twt = None  # Two-way time of seismic wave reflection.
        self.rgt = None  # Relative geologic time.
        self.info = {'basic':[], 
                     'vp':[], 
                     'rc': [],
                     'wavelet': [], 
                     'dipping': [],  
                     'folds': [], 
                     'faults': [], 
                     'meandering channels': [], 
                     'submarine channels': [], 
                     'noise': []}  # Model info.
        self.topo_out = None
        self.channel_out = None
        self.n_submarine = 0
        self.n_meandering = 0
        self.n_tributary = 0
        
        # Print model's basic information.
        if not mute:
            print('Model extent:')
            print('X range: %.2fm-%.2fm' % (self.Xmin, self.Xmax))
            print('Y range: %.2fm-%.2fm' % (self.Ymin, self.Ymax))
            print('Z range: %.2fm-%.2fm' % (self.Zmin, self.Zmax))
            print('Model resolution (XYZ): [%.2fm x %.2fm x %.2fm]' % (self.dX, self.dY, self.dZ))
            print('Model points (XYZ): [%d x %d x %d]' % (self.nX, self.nY, self.nZ))
        
        # Record model's basic information.
        self.info['basic'] = ['Model extent:\n', 
                              'X range: %.2fm-%.2fm\n' % (self.Xmin, self.Xmax), 
                              'Y range: %.2fm-%.2fm\n' % (self.Ymin, self.Ymax), 
                              'Z range: %.2fm-%.2fm\n' % (self.Zmin, self.Zmax), 
                              'Model resolution (XYZ): [%.2fm x %.2fm x %.2fm]\n' % (self.dX, self.dY, self.dZ),
                              'Model points (XYZ): [%d x %d x %d]\n' % (self.nX, self.nY, self.nZ)]
        

    def add_rc(self, rc_range=None, seed=None, mute=False):
        """
        Create a horizontally layered acoustic reflection coefficient (RC) model.
        
        :param rc_range: (List of floats) - Range of RC, in the format of [min, max] 
                                            Default values are [-1, 1].
        :param seed: (Integer) - The seed value needed to generate a random number. 
                                 Default value is None, which is to use the current system time.
        :param mute: (Bool) - If True, will not print anything. 
                              Default value is False.
        """
        # Control the random state.
        np.random.seed(seed)
        
        # Print progress.
        if not mute:
            sys.stdout.write('Generating RC model...')
        
        # Get RC range.
        if rc_range is None:
            rc_min, rc_max = -1, 1  # The default RC range.
        else:
            rc_min, rc_max = rc_range[0], rc_range[1]
        
        # Generate random numbers in RC range.
        rdm = (rc_max - rc_min) * np.random.random_sample((self.nZ,)) + rc_min
        
        # Initialize RC model with a 3d array filled with ones.
        self.rc = np.ones(shape=self.Z.shape, dtype=np.float32)
        
        # Generate model with random RC, each z-slice of the model has uniform RC. 
        for i in range(self.nZ):
            self.rc[:, :, i] = self.rc[:, :, i] * rdm[i]
        
        # Set RC of the first and last z-slices to 0.
        self.rc[:, :, 0] *= 0
        self.rc[:, :, -1] *= 0
        
        # Record RC information.
        self.info['rc'] = ['Acoustic reflection coefficient (RC):\n', 
                           'rc_range: %s\n' % rc_range, 
                           'seed: %s\n' % seed]
        # Print progress.
        if not mute:
            sys.stdout.write('Done.\n')

    
    def add_vp(self, 
               h_layer_list: list[tuple] = None, 
               fm_list: list[float] = [0.3, 0.6], 
               vp_list: list[tuple] = [(3000, 3500), 
                                       (3500, 5000), 
                                       (5000, 6500)], 
               vp_diff_list: list[tuple] = [(300, 500), 
                                            (300, 500), 
                                            (300, 500)], 
               vp_disturb: float = 100, 
               smooth: bool = True, 
               sigma: float = 2.0, 
               seed: int = None, 
               mute: bool = False):
        """
        Create horizontally layered seismic P-wave velocity (Vp) model.

        Args:
        h_layer_range (list of floats): Range of layer thickness [m].
                                        In the format of [min, max].
                                        The layer thickness will be randomly chosen within this range.
                                        Defaults to [0.1 * (Zmax - Zmin), 0.2 * (Zmax - Zmin)], 
                                        where Zmax is the maxima of the model's z-coordinates and
                                        Zmin is the minima of the model's z-coordinates.
        fm_list (list of floats): Lower boundaries of the shallow and middle formations.
                                  In the format of [shallow boundary, middle boundary]. 
                                  Each element in this list is a fraction between 0 and 1, 
                                  with 0 representing the model's top and 1 representing the model's bottom.
                                  Defaults to [0.3, 0.6].                              
        vp_list (list of tuples): Vp ranges of the shallow, middle and deep formations [m/s].
                                  In the format of [(min, max), (min, max), (min, max)].
                                  The Vp of each layer in the shallow, middle and deep formations
                                  will be randomly chosen within the respective ranges defined by those tuples.
                                  Defaults to [(3000, 3500), (3500, 5000), (5000, 6500)].                
        vp_diff_range (float): Vp difference range between two consecutive layers [m/s]. 
                               Defaults to [300, 500].              
        vp_disturb (float): Standard deviation of the random Vp disturbance in each layer [m/s]. 
                            Defaults to 100 m/s.
        smooth (bool): Whether to smooth the Vp model with a Gaussian filter. 
                       Defaults to True.
        sigma (float): Standard deviation of the Gaussian kernal. 
                       Defaults to 2.0.
                       Only effective when smooth is True.
        seed (int): Seed value to generate random numbers. 
                    Defaults to None, which is to use the current system time.
        mute (bool): Whether to mute printing. 
                     Defaults to False.
        """
        
        # Control the random state.
        np.random.seed(seed)
        random.seed(seed)
        
        # Print progress.
        if not mute:
            sys.stdout.write('\rGenerating Vp model...')
            
        # Initialization.
        self.vp = np.ones(self.Z.shape, dtype=np.float32)  # P-wave velocity.
        self.horizon = []
        
        depth_top = self.Zmin  # The initial layer's top depth.
        ind_bottom = 0  # The initial array Z-index of layer's bottom depth.
        if h_layer_list is None:
            h_layer_list = [(0.1 * (self.Zmax - self.Zmin), 0.2 * (self.Zmax - self.Zmin)), 
                             (0.1 * (self.Zmax - self.Zmin), 0.2 * (self.Zmax - self.Zmin)), 
                             (0.1 * (self.Zmax - self.Zmin), 0.2 * (self.Zmax - self.Zmin))]
        fmbt_idx = [int(fm_list[0] * (self.nZ - 1)),  # Array Z-index of the bottom of shallow formation.
                    int(fm_list[1] * (self.nZ - 1)),  # Array Z-index of the bottom of middle formation.
                    int(1.0 * (self.nZ - 1))]         # Array Z-index of the bottom of deep formation.
        
        # Save info.
        self.info['vp'] = ['P-wave velocity:\n',
                           'h_layer_list: %s\n' % h_layer_list, 
                           'fm_list: %s\n' % fm_list,   
                           'vp_list: %s\n' % vp_list, 
                           'vp_disturb: %s\n' % vp_disturb, 
                           'smooth: %s\n' % smooth]
        if smooth:
            self.info['vp'].append('sigma: %s\n' % sigma)
        self.info['vp'].append('seed: %s\n' % seed)
        
        # Assign velocity from the top to the bottom.
        vp_upper = 0
        for idx, vpf, hl, vpd in zip(fmbt_idx, vp_list, h_layer_list, vp_diff_list):
            fmbt = self.Zmin + idx * self.dZ  # Bottom depth of the formation.
            vpfmin, vpfmax = vpf  # Layer vp range of this formation.
            hlmin, hlmax = hl  # Layer thickness range in this formation.
            vpdmin, vpdmax = vpd  # Layer vp difference range in this formation.
            while ind_bottom < idx:
                # Set layer thickness randomly.
                h_layer = random.uniform(hlmin, hlmax)
                # Compute the layer's bottom depth.
                depth_bottom = depth_top + h_layer
                # Layer's bottom depth can not be greater than 125% formation bottom depth or Z_max.
                if depth_bottom > 1.25 * fmbt or depth_bottom > self.Zmax:
                    depth_bottom = min(1.25 * fmbt, self.Zmax)
                # Compute array Z-index of the layer's top and bottom depth.
                ind_top = int((depth_top - self.Zmin) // self.dZ)  # Layer's top depth.
                ind_bottom = int((depth_bottom - self.Zmin) // self.dZ)  # Layer's bottom depth.
                # Assign layer velocity.
                vp_diff = random.uniform(vpdmin, vpdmax)  # Vp difference with upper layer.
                vpl1 = vp_upper - vp_diff  # Lower vp.
                vpl2 = vp_upper + vp_diff  # Higher vp.
                if (vpl2 < vpfmin) or (vpl1 > vpfmax):  # Layer vp not in formation vp range, ignore vpdiff. 
                    vp = random.uniform(vpfmin, vpfmax)
                else:  # Layer vp in formation vp range.
                    if vpl1 < vpfmin:  # Lower vp out of formation vp range, use higher vp.
                        vp = vpl2
                    elif vpl2 > vpfmax:  # Higher vp out of formation vp range, use lower vp.
                        vp = vpl1
                    else:  # Randomly select lower or higher vp.
                        vp = random.choice([vpl1, vpl2])
                vp_upper = vp
                # Gaussian distributed velocity.
                if vp_disturb > 0.0:
                    shape = self.vp[:, :, ind_top:ind_bottom+1].shape
                    self.vp[:, :, ind_top:ind_bottom+1] = np.random.normal(loc=vp, scale=vp_disturb, size=shape)
                # Uniform velocity.
                else:
                    self.vp[:, :, ind_top:ind_bottom+1] *= vp
                # Z-coordinates of the horizon.
                hz = self.Zmin + (ind_bottom+1) * self.dZ  
                # Store the horizon.
                self.horizon.append(Horizon(z=hz, 
                                            vp=vp, 
                                            rgt=ind_bottom+1, 
                                            channel=0))
                # Update layer top depth.
                depth_top = hz
        
        # Smooth Vp.
        if smooth:
            for k in range(self.vp.shape[-1]):
                self.vp[:, :, k] = ndimage.gaussian_filter(self.vp[:, :, k], sigma=sigma)
        
        # Print progress.
        if not mute:
            sys.stdout.write(' Done.\n')

    
    def add_rgt(self, 
                mute: bool = False):
        """
        Create a relative geologic time (RGT) model.

        Args:
        mute (bool): Whether to mute printing. 
                     Defaults to False.
        """
        # Print progress.
        if not mute:
            sys.stdout.write("\rGenerating RGT model...")
        
        # Initialize RGT cube with ones.
        self.rgt = np.ones(self.Z.shape, dtype=np.float32)
        
        # Z-indexes.
        zid = np.arange(0, self.nZ, 1, dtype=np.float32)
        
        # RGT is 3D Z-indexes.
        for k in range(self.nZ):
            self.rgt[:, :, k] *= zid[k]
        
        # Print progress.
        if not mute:
            sys.stdout.write(" Done.\n")
    
    
    def smooth(self, param='all', sigma=1, mute=False):
        """
        Smooth 3D models with Gaussian filter.
        
        :param (String or list of strings): 3D models to be smoothed. For example, ['vp', 'channel'].
                                            1. 'vp' - Smooth Vp model.
                                            2. 'rgt' - Smooth RGT model.
                                            3. 'channel' - Smooth channel label.
                                            If you want to smooth all models, just assign param='all'.
                                            Defaults to 'all'. 
        :sigma (Float): Standard deviation of the Gaussian filter. 
                        Defaults to 1.
        """
        if param == 'all' or 'vp' in param:
            if not mute:
                sys.stdout.write('Smoothing Vp model...')
            if self.vp is None:
                raise ValueError("Vp model not found. \
                                 Use 'add_vp' function to create a Vp model.")
            self.vp = ndimage.gaussian_filter(self.vp, sigma=sigma)
            if not mute:
                sys.stdout.write(' Done.\n')
            
        if param == 'all' or 'rgt' in param:
            if not mute:
                sys.stdout.write("Smoothing RGT model...")
            if self.rgt is None:
                raise ValueError("RGT model not found. \
                                  Use 'add_rgt' function to create a RGT model.")
            self.rgt = ndimage.gaussian_filter(self.rgt, sigma=sigma)
            if not mute:
                sys.stdout.write(" Done.\n")
        
        if param == 'all' or 'channel' in param:
            if not mute:
                sys.stdout.write('Smoothing channel label...')
            if self.channel is None:
                raise ValueError("River channel label not found. \
                                 Use 'add_meandering_channel_from_database', 'add_tributary_channel_from_database', \
                                 or 'add_submarine_channel' to create channel label.")
            self.channel = self.channel.astype(np.float32)
            self.channel = ndimage.gaussian_filter(self.channel, sigma=sigma)
            if not mute:
                sys.stdout.write(' Done.\n')
        
    
    def compute_Ip(self, rho=2.4):
        """
        Compute seismic P-wave impedance in terms of constant rock density.
        
        :rho (float): Rock density. Defaults to 2.4.
        """
        if self.vp is None:
            raise ValueError("P-wave velocity (Vp) model is not exist. Use 'add_vp' function to create a Vp model.")
        self.Ip = self.vp * rho
        
    
    def compute_rc(self, mute=False):
        """
        Compute reflection coefficients (RC) in terms of constant rock density.
        RC = (Vp[i+1] - Vp[i]) / (Vp[i+1] + Vp[i])
        
        :param mute: (Bool) - If True, will not print anything. 
                              Default value is False.
        """
        # If RC model does not exist.
        if self.rc is None:
            if self.vp is None:
                raise ValueError("P-wave velocity (Vp) model is not exist. Use 'add_vp' function to create a Vp model.")
            self.rc = np.zeros(self.vp.shape, dtype=np.float32)
            for i in range(self.rc.shape[2] - 1):
                if not mute:
                    sys.stdout.write('\rComputing reflection coefficient: %.2f%%' % ((i + 1) / (self.vp.shape[2] - 1) * 100))
                self.rc[:, :, i] = (self.vp[:, :, i + 1] - self.vp[:, :, i]) / (self.vp[:, :, i + 1] + self.vp[:, :, i])
            if not mute:
                sys.stdout.write('\n')
        
        # If RC model exists.
        else:
            s = input('A reflection coefficient model exists, do you want to overwrite it? [Y/N]')
            if s.lower() == 'y':
                if self.vp is None:
                    raise ValueError("P-wave velocity (Vp) model is not defined. Use 'add_vp' function to define Vp model.")
                for i in range(self.rc.shape[2] - 1):
                    if not mute:
                        sys.stdout.write(
                            '\rComputing reflection coefficient: %.2f%%' % ((i + 1) / (self.vp.shape[2] - 1) * 100))
                    self.rc[:, :, i] = (self.vp[:, :, i + 1] - self.vp[:, :, i]) / (self.vp[:, :, i + 1] + self.vp[:, :, i])
                if not mute:
                    sys.stdout.write('\n')
            else:
                print('Reflection coefficient model is unchanged.')

    
    def make_synseis(self, 
                     wavelet_type: str = 'ricker', 
                     A: float = 1.0, 
                     f_ricker: float = 30, 
                     f_ormsby: tuple[float] = (5, 10, 40, 45), 
                     dt: float = 0.002, 
                     length: float = 0.1,
                     mark_channel: bool = True, 
                     plot_wavelet: bool = True, 
                     mute: bool = False):
        """
        Make synthetic seismic data.
        
        Args:
        wavelet_type (str): Wavelet type.
                            Options: 1. 'ricker' - Ricker wavelet.
                                     2. 'ormsby' - Ormsby wavelet.
                            Default value is 'ricker'.
        A (float): Maximum amplitude of the wavelet. 
                   Defaults to 1.0.
        f_ricker (float): Peak frequency of the Ricker wavelet (unit: Hz). 
                          Default value is 30Hz.
                          Only effective when 'type' is 'ricker'.
        f_ormsby (tuple of float): Four frequencies of the Ormsby wavelet (unit: Hz), 
                                   in the format of (low-cut, low-pass, high-pass, high-pass).
                                   Default values are (5, 10, 40, 45)Hz.
        dt (float): Sampling time interval of the wavelet (unit: s). 
                    Defaults to 0.002s.
        length (float): Time duration of the wavelet (unit: s). 
                        Defaults to 0.1s.
        mark_channel (str): Mark the seismic response of channels.
        plot_wavelet (bool): Whether to visualize the wavelet. 
                             Defaults to True.
        mute (bool): Whether to mute printing.
                     Defaults to True.
        """
        # Check if RC model exists.
        if self.rc is None:
            raise ValueError("Reflection coefficient (RC) model is not defined. Use 'add_rc' or 'compute_rc' functions to define RC model.")
        
        # Initialize the output.
        self.seismic = np.zeros(shape=self.rc.shape, dtype=np.float32)
        
        # Initialize seismic label (seismic response marker of channels).
        if mark_channel:
            if self.channel is None:
                raise ValueError("Channel label not found.")
            else:
                if self.channel.dtype == np.int16:
                    label_type = 'instance'
                    self.seis_label = np.zeros(self.seismic.shape, dtype=np.int16)
                else:
                    label_type = 'semantic'
                    self.seis_label = np.zeros(self.seismic.shape, dtype=np.uint8)
                # Add padding to the channel labels, otherwise their seismic labels will be problematic.
                channelp = padding3D(self.channel, pad_up=50, pad_down=50)
                self.seis_label = padding3D(self.seis_label, pad_up=50, pad_down=50)
        
        # Generate seismic wavelet.
        t = np.arange(-length / 2, length / 2, dt, dtype=np.float32)
        if wavelet_type == 'ricker':  # Riker wavelet.
            wavelet = (1 - 2 * math.pi ** 2 * f_ricker ** 2 * t ** 2) * np.exp(-math.pi ** 2 * f_ricker ** 2 * t ** 2)
            wavelet *= A
        if wavelet_type == 'ormsby':  # Ormsby wavelet.
            f1, f2, f3, f4 = f_ormsby
            wavelet = (math.pi * f1 ** 2) / (f2 - f1) * np.sinc(math.pi * f1 * t) ** 2 - \
                      (math.pi * f2 ** 2) / (f2 - f1) * np.sinc(math.pi * f2 * t) ** 2 - \
                      (math.pi * f3 ** 2) / (f4 - f3) * np.sinc(math.pi * f3 * t) ** 2 + \
                      (math.pi * f4 ** 2) / (f4 - f3) * np.sinc(math.pi * f4 * t) ** 2
            wavelet *= A
        
        # Record wavelet info.
        self.info['wavelet'] = ['Seismic Wavelet:\n',
                                'wavelet_type: %s\n' % wavelet_type,  
                                'A: %s\n' % A, 
                                'length: %s(s)\n' % length, 
                                'dt: %s(s)\n' % dt]
        if wavelet_type == 'ricker':
            self.info['wavelet'].append('f_ricker: %s\n' % f_ricker)
        if wavelet_type == 'ormsby':
            self.info['wavelet'].append('f_ormsby: %s\n' % f_ormsby)
        
        # Compute amplitude spectrum of the wavelet.
        n = len(wavelet)
        xf = fftfreq(n, dt)[:n//2]
        yf = np.abs(fft(wavelet))[:n//2]
        
        # Make synthetic seismic data.
        for i in range(self.rc.shape[0]):
            for j in range(self.rc.shape[1]):
                # Print progress.
                if not mute:
                    sys.stdout.write('\rGenerating synthetic seismic data: %.2f%%' %
                                    ((i*self.rc.shape[1]+j+1) / (self.rc.shape[0]*self.rc.shape[1]) * 100))
                
                # Seismic data.
                self.seismic[i, j, :] = np.convolve(self.rc[i, j, :], wavelet, mode='same')
                
                # Mark the seismic response of channel.
                # Semantic label. 
                # (0: non-channel, 1: channel)
                if mark_channel and label_type == 'semantic':
                    clb = channelp[i, j, :].copy()
                    # Skip this trace if the labels are all zeros.
                    if (clb == 0).all():
                        continue
                    else:
                        # Convolve wavelet and channel label.
                        slb = np.convolve(clb, wavelet, mode='same')
                        # Compute envelope.
                        slb = np.abs(hilbert(slb))
                        # Ensure the channel fill has the strongest seismic response.
                        slb = np.maximum(slb, clb)  
                        # Binarize.
                        slb[slb < 0.5] = 0  
                        slb[slb >= 0.5] = 1
                        # Update seismic label.
                        self.seis_label[i, j, :] = slb
                
                # Instance label.
                # (101: meandering channel 1, 102: meandering channel 2, ...)
                # (201: tributary channel 1, 202: tributary channel 2, ...)
                # (301: submarine channel 1, 302: submarine channel 2, ...)
                if mark_channel and label_type == 'instance':
                    # Get unique label values in this trace.
                    lbl = np.unique(channelp[i, j, :])
                    if (lbl == 0).all():
                        # Skip this trace if the labels are all zeros.
                        continue
                    else:
                        # Rule out non-channel.
                        lbl = lbl[lbl > 0]
                    for k in range(len(lbl)):
                        # A trace in channel label cube.
                        clb = channelp[i, j, :].copy() 
                        # Convert to binary.
                        clb[clb != lbl[k]] = 0
                        clb[clb == lbl[k]] = 1
                        # Convolve wavelet and channel label.
                        slb = np.convolve(clb, wavelet, mode='same')
                        # Compute the envelope. 
                        slb = np.abs(hilbert(slb))
                        # Ensure the channel fill has the strongest seismic response.
                        slb = np.maximum(slb, clb)  
                        # Binarize.
                        slb[slb < 0.5] = 0  
                        slb[slb >= 0.5] = 1
                        # Assign instance id.
                        slb *= lbl[k]
                        # Update.
                        self.seis_label[i, j, :] = np.maximum(self.seis_label[i, j, :], slb)
                        
        # Depadding the seismic label.
        self.seis_label = depadding3D(self.seis_label, pad_up=50, pad_down=50)
        
        if not mute:
            sys.stdout.write('\n')
        
        # Visualize wavelet.
        if plot_wavelet:
            fig, ax = plt.subplots(1, 2, figsize=(16, 4))
            ax[0].plot(t, wavelet, 'k', lw=3)
            ax[0].set_xlabel('Time(s)', fontsize=14)
            ax[0].set_ylabel('Amplitude', fontsize=14)
            ax[0].tick_params(labelsize=12)
            ax[1].plot(xf, yf, 'k', lw=3)
            ax[1].set_xlabel('Frequency(Hz)', fontsize=14)
            ax[1].set_ylabel('Amplitude', fontsize=14)
            ax[1].tick_params(labelsize=12)
            fig.tight_layout()
            plt.show()

    
    def add_dipping(self, 
                    a_range: list[float] = [0.1, 0.3], 
                    b_range: list[float] = [0.1, 0.3], 
                    Xc_range: list[float] = [0.4, 0.6], 
                    Yc_range: list[float] = [0.4, 0.6], 
                    seed: int = None, 
                    mute: bool = False):
        """
        Simulate dipping structure.
        
        Args:
        a_range (list of float): Range of 'a', which determines the dipping angle in x-direction, 
                                 in the format of [min, max].
                                 Default range is [0.1, 0.3].
                                 Larger value means more dipping.
        b_range (list of float): Range of 'b', which determines the dipping angle in y-direction, 
                                 in the format of [min, max].
                                 Default range is [0.1, 0.3].
                                 Larger value means more dipping.
        Xc_range (list of float): X-coordinate range of the dipping center, in the format of [min, max].
                                  This range must be between 0 and 1, with 0 means the minimum x-coordinate and 1 means
                                  the maximum x-coordinate. 
                                  Default range is [0.4, 0.6].
        Yc_range (list of float): Y-coordinate range of the dipping center, in the format of [min, max].
                                  This range must be between 0 and 1, with 0 means the minimum y-coordinate and 1 means
                                  the maximum y-coordinate.
                                  Default range is [0.4, 0.6].
        seed (int): The seed value needed to generate a random number. 
                    Defaults to None, which is to use the current system time.
        mute (bool): If True, will not print anything. 
                     Defaults to False.
        """
        # Random state control.
        random.seed(seed)
        
        # Print progress.
        if not mute:
            sys.stdout.write('\rSimulating dipping structure...')
        
        # Initialize the dipping parameter table.
        tb = PrettyTable()
        tb.field_names = ['a', 'b', 'Xc(m)', 'Yc(m)']
        tb.float_format = '.2'
            
        # Record input parameters.
        self.info['dipping'] = ['Dipping:\n', 
                                'a_range: %s\n' % a_range, 
                                'b_range: %s\n' % b_range, 
                                'Xc_range: %s\n' % Xc_range, 
                                'Yc_range: %s\n' % Yc_range, 
                                'seed: %s\n' % seed]
        
        # Vertical shift field.
        a = random.uniform(a_range[0], a_range[1]) * random.choice([-1, 1])
        b = random.uniform(b_range[0], b_range[1]) * random.choice([-1, 1])
        Xc = random.uniform(Xc_range[0], Xc_range[1]) * (self.Xmax - self.Xmin) + self.Xmin
        Yc = random.uniform(Yc_range[0], Yc_range[1]) * (self.Ymax - self.Ymin) + self.Ymin
        c0 = -a * Xc - b * Yc
        dZ = a * self.X + b * self.Y + c0
        
        # Shift Z-coordinates.
        self.Z -= dZ
        
        # Dipping parameter table.
        tb.add_row([a, b, Xc, Yc])
        self.info['dipping'].append(tb)  # Record the table. 
        
        # Print.
        if not mute:
            sys.stdout.write(' Done.\n')
            print(tb)
    
    
    def add_fold(self, 
                 N: int = 10, 
                 miu_X_range: list[float] = [0, 1], 
                 miu_Y_range: list[float] = [0, 1], 
                 sigma_range: list[float] = [0.1, 0.2], 
                 A_range: list[float] = [0.1, 0.2],
                 d_fold: float = 300,  
                 zscale: float = 1.5, 
                 sync: bool = False, 
                 seed: int = None, 
                 mute: bool = False):
        """
        Simulate folds with a set of Gaussian functions.
        
        N (int): The number of Gaussian functions.
                 This is also the approximate number of folds.
                 Defaults to 10.
        miu_X_range (list of float): X-coordinate range of the fold center (i.e. mean of the Gaussian function).
                                     This range must be between 0 and 1, with 0 means the minimum x-coordiante and
                                     1 means the maximum x-coordinate.
                                     The default range is [0, 1].
        miu_Y_range (list of float): Y-coordinate range of the fold center (i.e. mean of the Gaussian function).
                                     This range must be between 0 and 1, with 0 means the minimum y-coordinate and
                                     1 means the maximum y-coordinate.
                                     The default range is [0, 1].
        sigma_range (list of float): Half-width range of the fold (i.e. standard deviation of the Gaussian function).
                                     This range must be between 0 and 1, with 0 means the minimum horizontal coordinate
                                     and 1 means the maximum horizontal coordinate.
                                     The default range is [0.1, 0.2].
        A_range (list of float): Amplitude range of the fold (i.e. amplitude of the Gaussian function).
                                 This range must be between 0 and 1, with 0 means the minimum z-coordinate and 1 means
                                 the maximum z-coordinate.
                                 The default range is [0.1, 0.2].
        d_fold (float): Minimum distance between each pair of folds [m].
                        Defaults to 300 m.
        z_scale (float): Scaling factor of fold amplitude.
                         Defaults to 1.5.
        sync (bool): Whether the fold bending will not increase with depth.
                     Defaults to False.
        seed (int): The seed value needed to generate a random number. 
                    Defaults to None, which is to use the current system time.
        mute (bool): Whether to mute printing verbose info.
                     Defaults to False  
        """
        # Random state control.
        random.seed(seed)
        
        # Print progress. 
        if not mute:
            sys.stdout.write('\rSimulating folds...')
        
        # Initialize the fold parameter table.
        fold_parameter = PrettyTable()
        fold_parameter.field_names = ['Number', 'miu_X (m)', 'miu_Y (m)', 'sigma (m)', 'Amplitude (m)']
        fold_parameter.float_format = '.2'
        
        # Save info.
        self.info['folds'] = ['Folds:\n', 
                              'miu_X_range: %s\n' % miu_X_range, 
                              'miu_Y_range: %s\n' % miu_Y_range, 
                              'sigma_range: %s\n' % sigma_range, 
                              'A_range: %s\n' % A_range, 
                              'Sync: %s\n' % sync, 
                              'seed: %s\n' % seed]
        
        # Create folds.
        Gaussian_sum = 0  # Initialize the summation of Gaussian functions.
        x0y0_list = []  # A list of all fold center coordinates.
        for i in range(N):
            flag = 1
            while(flag == 1):
                miu_X = random.uniform(miu_X_range[0], miu_X_range[1]) * (self.Xmax - self.Xmin) + self.Xmin
                miu_Y = random.uniform(miu_Y_range[0], miu_Y_range[1]) * (self.Ymax - self.Ymin) + self.Ymin
                if len(x0y0_list) == 0:
                    x0y0_list.append((miu_X, miu_Y))
                    flag = 0
                else:
                    for item in x0y0_list:
                        x0, y0 = item
                        d = math.sqrt((miu_X - x0)**2 + (miu_Y - y0)**2)
                        if d >= d_fold:
                            x0y0_list.append((miu_X, miu_Y))
                            flag = 0
            sigma = random.uniform(sigma_range[0], sigma_range[1]) * min(self.Xmax - self.Xmin, self.Ymax - self.Ymin)
            A = random.uniform(A_range[0], A_range[1]) * (self.Zmax - self.Zmin)
            # The Gaussian function.
            f_Gaussian = A * np.exp(-1 * ((self.X - miu_X) ** 2 + (self.Y - miu_Y) ** 2) / (2 * sigma ** 2))
            Gaussian_sum += f_Gaussian  # Combine the Gaussian functions.
            fold_parameter.add_row([i + 1, miu_X, miu_Y, sigma, A])  # Visualizing parameters.
        
        # Shift the Z-coordinates vertically.
        if sync is False:
            self.Z = self.Z - zscale * self.Z / self.Z.max() * Gaussian_sum
        else:
            self.Z = self.Z - Gaussian_sum
                
        # Save info.
        self.info['folds'].append(fold_parameter)
        
        # Print.
        if not mute:
            sys.stdout.write('Done.\n')
            print(fold_parameter)

    
    def add_fault(self, N=3, reference_point_range=None, phi_range=None, theta_range=None,
                  d_max_range=None, lx_range=None, ly_range=None, gamma_range=None, beta_range=None,
                  curved_fault=False, n_perturb=20, perturb_range=None, d_fault=300.0,
                  computation_mode='parallel', seed=None, mute=False):
        """
        Simulate faults.
        
        :param N: (Integer) - The number of faults. 
                              Default value is 3.
        :param reference_point_range: (List of floats) - Coordinate ranges of the fault plane center (unit: m), in the format of [X0_min, X0_max, Y0_min, Y0_max, Z0_min, Z0_max].
                                                         Must be decimals between 0 and 1.
                                                         For the center of each fault plane:
                                                         The actual x-coordinate will be randomly chosen between:
                                                         'X0_min * (Xmax - Xmin)' and 'X0_max * (Xmax - Xmin)'.
                                                         The actual y-coordinate will be randomly chosen between:
                                                         'Y0_min * (Ymax - Ymin)' and 'Y0_max * (Ymax - Ymin)'.
                                                         The actual z-coordinate will be randomly chosen between:
                                                         'Z0_min * (Zmax - Zmin)' and 'Z0_max * (Zmax - Zmin)'.
                                                         Default values are [0.1, 0.9, 0.1, 0.9, 0.3, 0.9].
        :param phi_range: (List of floats) - Strike angle range (unit: degree), in the format of [min, max].
                                             The strike angle of each fault will be randomly chosen between this range.
                                             Default values are [0, 360].
        :param theta_range: (List of floats) - Dip angle range (unit: degree), in the format of [min, max].
                                               The dip angle of each fault will be randomly chosen between this range. 
                                               Default values are [0, 90].
        :param d_max_range: (List of floats) - Range of maximum displacement on the fault plane (unit: meter), in the format of [min, max].
                                               Must be two decimals no less than zero.
                                               For each fault, the actual displacement value will be randomly chosen between:
                                               'min * y_range' and 'max * y_range', where y_range is the model's extent in dip direction of the fault.
                                               Default values are [0.1, 0.3].
        :param lx_range: (List of floats) - Range of axial length of the elliptical fault displacement field in strike direction (unit: meter), in the format of [min, max].
                                            Must be two decimals no less than zero.
                                            For each fault, the actual axial length will be randomly chosen between:
                                            'min * x_range' and 'max * x_range', where x_range is the model's extent in strike direction of the fault.
                                            Default values are [0.5, 1.0]. 
        :param ly_range: (List of floats) - Range of axial length of the elliptical fault displacement field in dip direction (unit: meter), in the format of [min, max].
                                            Must be two decimals no less than zero.
                                            For each fault, the actual axial length will be randomly chosen between:
                                            'min * y_range' and 'max * y_range', where y_range is the model's extent in dip direction of the fault.
                                            Default values are [0.1, 0.5].
        :param gamma_range: (List of floats) - Range of reverse drag radius (unit: m), in the format of [min, max].
                                               Must be two decimals no less than zero.
                                               For each fault, the actual radius will be randomly chosen between:
                                               'min * z_range' and 'max * z_range', where z_range is the model's extent in normal direction of the fault.
                                               Default values are [0.1, 0.5].
        :param beta_range: (List of floats) - Range of 'hanging-wall displacement / d_max', in the format of [min, max].
                                              The beta value of each fault will be randomly chosen between this range.
                                              Default values are [0.5, 1.0].
        :param curved_fault: (Bool) - Whether to create a curved fault. 
                                      Default value is False.
        :param n_perturb: (Integer) - Number of perturbation points near the fault plane, which are used to create a curved fault plane. 
                                      Default value is 20.
                                      Only effective when 'curved_fault' is True.
        :param perturb_range: (List of floats) - Normal distance range from perturbation points to the fault plane (unit: meter), in the format of [min, max].
                                                 Must be two decimals between -1 and 1.
                                                 For each fault, the actual distance will be randomly chosen between:
                                                 'min * z_range' and 'max * z_range', where z_range is the model's extent in normal direction of the fault.
                                                 Default values are [-0.05, 0.05].
                                                 Only effective when 'curved_fault' is True.
        :param d_fault: (Float) - Minimum distance between faults (unit: meter).
                                  Default value is 300.0.
        :param computation_mode: (String) - The computation mode.
                                            Options are:
                                            1. 'parallel' - which is to break down the model's coordinate arrays into
                                                            slices and simulate curved faults in parallel.
                                            2. 'non-parallel' - takes the whole coordinate arrays as input to simulate
                                                                the curved faults.
                                            Notice that when the model size is small (e.g. 32 x 32 x 32), taking the
                                            whole coordinate array as input will be faster.
                                            In addition, when the memory is not enough, using the 'parallel' mode
                                            may solve the problem.
        :param seed: (Integer) - The seed value needed to generate a random number. 
                                 Default value is None, which is to use the current system time.
        :param mute: (Bool) - If True, will not print anything. 
                              Default value is False. 
        """
        # Random state control.
        random.seed(seed)
        np.random.seed(seed)
        
        # Print.
        if curved_fault:  # Curved fault.
            if computation_mode != 'parallel' and computation_mode != 'non-parallel':
                raise ValueError("'computation_mode' must be 'parallel' or 'non-parallel'.")
            else:
                if not mute:
                    sys.stdout.write(f'\rSimulating curved fault in {computation_mode} mode...')
        else:  # Planar fault.
            if not mute:
                sys.stdout.write('\rSimulating planar fault...')
        
        # Initialize fault marker.
        self.fault = np.zeros(self.Z.shape, dtype=np.int32)  
        
        # Initialize the fault parameter table.
        fault_parameter = PrettyTable()
        fault_parameter.field_names = ['Fault Number', 'X0(m)', 'Y0(m)', 'Z0(m)', 
                                       'phi(degree)', 'theta(degree)', 
                                       'dmax(m)', 'lx(m)', 'ly(m)',
                                       'gamma', 'beta']
        fault_parameter.float_format = '.2'
        
        # Assign default parameters.
        if reference_point_range is None:
            reference_point_range = [0.1, 0.9, 0.1, 0.9, 0.3, 0.9]
        if phi_range is None:
            phi_range = [0, 360]
        if theta_range is None:
            theta_range = [0, 90]
        if lx_range is None:
            lx_range = [0.5, 1.0]
        if ly_range is None:
            ly_range = [0.1, 0.5]
        if d_max_range is None:
            d_max_range = [0.1, 0.3]
        if perturb_range is None:
            perturb_range = [-0.05, 0.05]
        if gamma_range is None:
            gamma_range = [0.1, 0.5]
        if beta_range is None:
            beta_range = [0.5, 1]
        
        # Save info.
        self.info['faults'] = ['Faults:\n',
                               'reference_point_range: %s\n' % reference_point_range,
                               'phi_range: %s\n' % phi_range, 
                               'theta_range: %s\n' % theta_range, 
                               'lx_range: %s\n' % lx_range, 
                               'ly_range: %s\n' % ly_range, 
                               'd_max_range: %s\n' % d_max_range, 
                               'gamma_range: %s\n' % gamma_range, 
                               'beta_range: %s\n' % beta_range, 
                               'curved_fault: %s\n' % curved_fault]
        if curved_fault:
            self.info['faults'].append('n_perturb: %s\n' % n_perturb)
            self.info['faults'].append('perturb_range: %s\n' % perturb_range)
            self.info['faults'].append('computation_mode: %s\n' % computation_mode)
        self.info['faults'].append('seed: %s\n' % seed)
        
        # Create faults.
        X0Y0_list = []  # Save X- and Y-coordinates of the fault centers.
        for n in range(N):
            cnt = 1
            while(cnt > 0):
                # Fault center coordinates.
                X0, Y0, Z0 = random.uniform(reference_point_range[0], reference_point_range[1]) * (self.Xmax - self.Xmin) + self.Xmin, \
                             random.uniform(reference_point_range[2], reference_point_range[3]) * (self.Ymax - self.Ymin) + self.Ymin, \
                             random.uniform(reference_point_range[4], reference_point_range[5]) * (self.Zmax - self.Zmin) + self.Zmin
                cnt = 0  # Number of fault center whose distance to the new generated fault center is shorter than the minimum distance.
                if len(X0Y0_list):
                    for X0Y0 in X0Y0_list:
                        X, Y = X0Y0  # Coordinates of previous fault centers.
                        d = math.sqrt((X - X0) ** 2 + (Y - Y0) ** 2)
                        if d < d_fault:
                            cnt += 1
            X0Y0_list.append((X0, Y0))
            phi = random.uniform(phi_range[0], phi_range[1])
            theta = random.uniform(theta_range[0], theta_range[1])
            phi, theta = [math.radians(phi), math.radians(theta)]  # Conversion from degree to radian.
            R = [[math.sin(phi), - math.cos(phi), 0],  # Rotation matrix.
                 [math.cos(phi) * math.cos(theta), math.sin(phi) * math.cos(theta), math.sin(theta)],
                 [math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), -math.cos(theta)]]
            R = np.array(R, dtype=np.float32)
            cor_g = np.array([(self.X - X0).ravel(order='C'),  # Coordinate array of the model, with (X0, Y0, Z0) as center.
                              (self.Y - Y0).ravel(order='C'),
                              (self.Z - Z0).ravel(order='C')], dtype=np.float32)
            # Coordinate transformation (Cartesian -> fault plane).
            # 'x' is the strike direction, 'y' is the dip direction and 'z' is the normal direction.
            [x, y, z] = R @ cor_g
            x = x.reshape(self.X.shape, order='C')
            y = y.reshape(self.Y.shape, order='C')
            z = z.reshape(self.Z.shape, order='C')
            lx = random.uniform(lx_range[0] * (x.max() - x.min()), lx_range[1] * (x.max() - x.min()))
            ly = random.uniform(ly_range[0] * (y.max() - y.min()), ly_range[1] * (y.max() - y.min()))
            r = np.sqrt((x / lx) ** 2 + (y / ly) ** 2)  # The elliptical surface along the fault plane.
            r[r > 1] = 1  # To make the displacement = 0 outside the elliptical surface.
            # The elliptical displacement field along the fault plane.
            d_max = random.uniform(d_max_range[0] * (y.max() - y.min()), d_max_range[1] * (y.max() - y.min()))
            d = 2 * d_max * (1 - r) * np.sqrt((1 + r) ** 2 / 4 - r ** 2)
            f = 0  # Define fault plane (0 for planar surface).
            # Create curved fault plane.
            if curved_fault:
                # Randomly choose the 3-D coordinates of perturbation points.
                perturb_range = [perturb_range[0] * (z.max() - z.min()), perturb_range[1] * (z.max() - z.min())]
                x_perturb = (x.max() - x.min()) * np.random.random_sample((n_perturb,)) + x.min()
                y_perturb = (y.max() - y.min()) * np.random.random_sample((n_perturb,)) + y.min()
                z_perturb = \
                    (perturb_range[1] - perturb_range[0]) * np.random.random_sample((n_perturb,)) + perturb_range[0]
                # Use the perturbation points to calculate the parameters of Bi-harmonic Spline interpolator.
                interpolator = BiharmonicSpline3D(x_perturb, y_perturb, z_perturb)
                # Interpolate a curved fault plane.
                if computation_mode == 'parallel':
                    n_cores = multiprocessing.cpu_count()  # Get the number of cpu cores.
                    f = Parallel(n_jobs=n_cores)(delayed(compute_f_parallel)(i, x, y, interpolator)
                                                 for i in range(x.shape[0]))  # Compute in parallel.
                    f = np.array(f, dtype=np.float32)
                else:
                    f = interpolator(x, y)
            # Mark faults.
            z_resolution = (z.max() - z.min()) / (self.nZ - 1)
            ind = (np.abs(z - f) < z_resolution) & (d > 0)
            self.fault[ind] = 1
            # Nonlinear scalar function that decreases in normal direction from fault plane.
            gamma = random.uniform(gamma_range[0] * (z.max() - z.min()), gamma_range[1] * (z.max() - z.min()))
            alpha = (1 - np.abs(z - f) / gamma) ** 2
            # Initialize the displacement array.
            Dx = 0  # Strike displacement.
            Dy = np.zeros(shape=y.shape, dtype=np.float32)  # Dip displacement.
            Dz = 0  # Normal displacement.
            # Calculate volumetric displacement of the hanging-wall.
            beta = random.uniform(beta_range[0], beta_range[1])
            Dy[(z > f) & (z <= f + gamma)] = beta * d[(z > f) & (z <= f + gamma)] * alpha[(z > f) & (z <= f + gamma)]
            # Calculate volumetric displacement of the foot-wall.
            Dy[(z >= f - gamma) & (z <= f)] = \
                (beta - 1) * d[(z >= f - gamma) & (z <= f)] * alpha[(z >= f - gamma) & (z <= f)]
            # Add fault displacement.
            x = x + Dx
            y = y + Dy
            if curved_fault:
                if computation_mode == 'parallel':
                    Dz = Parallel(n_jobs=n_cores)(delayed(compute_Dz_parallel)(i, x, y, f, interpolator)
                                                  for i in range(x.shape[0]))  # Compute in parallel.
                    Dz = np.array(Dz, dtype=np.float32)
                else:
                    Dz = interpolator(x, y) - f
            z = z + Dz
            # Coordinate transformation (fault plane -> Cartesian).
            cor_f = np.array([x.ravel(order='C'),  # Coordinate array of the model.
                              y.ravel(order='C'),
                              z.ravel(order='C')], dtype=np.float32)
            [X_faulted, Y_faulted, Z_faulted] = np.linalg.inv(R) @ cor_f + np.array([[X0], [Y0], [Z0]], dtype=np.float32)
            self.X = X_faulted.reshape(self.X.shape, order='C')
            self.Y = Y_faulted.reshape(self.Y.shape, order='C')
            self.Z = Z_faulted.reshape(self.Z.shape, order='C')
            # Update fault parameter table.
            fault_parameter.add_row([n + 1, X0, Y0, Z0, phi * 180 / math.pi, theta * 180 / math.pi,
                                     d_max, lx, ly, gamma, beta])
        
        # Save info.
        self.info['faults'].append(fault_parameter)
        
        # Print.
        if not mute:
            sys.stdout.write('Done.\n')
            print(fault_parameter)
            
    
    def add_meandering_channel(self, 
                               N: int = 1,
                               X_pos_range: list[float] = [0.0, 1.0],
                               Y_pos_range: list[float] = [0.2, 0.8], 
                               Z_pos_range: list[float] = [0.0, 1.0], 
                               strike_range: list[float] = [0, 360],    
                               W_range: list[float] = [150, 300], 
                               D_range: list[float] = [15, 20],
                               Wx_range: list[float] = [150, 300], 
                               Dx_range: list[float] = [15, 20],  
                               kl_range: list[float] = [40, 50], 
                               Cf_range: list[float] = [0.05, 0.08],
                               n_iter_range: list[int] = [500, 1000], 
                               n_bends_range: list[int] = [30, 50], 
                               perturb_range: list[float] = [0.02, 0.05],
                               vpfill_range: list[float] = [2500, 3000],
                               epsilon_range: list[float] = [0, 1],  
                               kv: float = 3, 
                               aggr: float = 4, 
                               t_incision: list = [[0.1, 0.2], [0.3, 0.5]],
                               t_aggradation: list = [[0.7, 0.8], [0.9, 1.0]],
                               incision_mode: int|str = 'random', 
                               delta_s: float = None, 
                               dt: float = 0.1, 
                               pad_up: int = 0, 
                               pad_down: int = 0, 
                               save_iter: int = 20,
                               co_offset: int = 20,
                               show_migration: bool = False,  
                               seed: int = None,
                               instance_label: bool = True,  
                               mute: bool = False):
        
        # Set random seed value.
        random.seed(seed)
        np.random.seed(seed)
        
        if self.channel_out is None:
            self.channel_out = []
        
        # Check Vp model.
        if self.vp is None:
            raise ValueError("Vp model not found.")
        
        # Initialize channel markers.
        if self.channel is None:
            if instance_label:
                # Instance label.
                # 101: channel 1, 102: channel 2, 103: channel 3, ...
                self.channel = np.zeros(self.Z.shape, dtype=np.int16)
            else:
                # Semantic label.
                # 0: non-channel, 1: channel.
                self.channel = np.zeros(self.Z.shape, dtype=np.uint8)

        # Initialize table.
        meander_parameter = PrettyTable()
        meander_parameter.field_names = ['Number', 'X (m)', 'Y (m)', 'Z (m)', 'strike (degree)', 'width (m)', 
                                         'depth (m)', 's_init (m)', 'n_bends', 'iteration', 'dt (year)', 
                                         'migration rate constant (m/year)','friction factor', 'vp_fill (m/s)']
        meander_parameter.float_format = '.2'
        
        # Horizon indices.
        if self.horizon is None:
            raise ValueError("Horizon not found.")
        else:
            horizon_indices = list(range(len(self.horizon)))
        
        # Get horizon id in z range.
        zmin = self.Zmin + Z_pos_range[0] * (self.Zmax - self.Zmin)
        zmax = self.Zmin + Z_pos_range[1] * (self.Zmax - self.Zmin)
        horizon_in_range = []
        for i in horizon_indices:
            if zmin <= self.horizon[i].z <= zmax:
                horizon_in_range.append(i)
        
        # Simulation begins.
        for n in range(N):
            
            # Count channels.
            self.n_meandering += 1
            if self.n_meandering > 99:  # Maximum 99 channels.
                break
            
            # Print progress.
            if not mute:
                print("Simulating meandering channel [%d/%d]" % ((n+1), N))    
            
            # Assign parameters.
            strike = random.uniform(strike_range[0], strike_range[1])
            n_bends = random.randint(n_bends_range[0], n_bends_range[1])
            n_iter = random.randint(n_iter_range[0], n_iter_range[1])
            W = random.uniform(W_range[0], W_range[1])
            D = random.uniform(D_range[0], D_range[1])
            Wx = random.uniform(Wx_range[0], Wx_range[1])
            Dx = random.uniform(Dx_range[0], Dx_range[1])
            kl = random.uniform(kl_range[0], kl_range[1])
            Cf = random.uniform(Cf_range[0], Cf_range[1])
            vpfill = random.uniform(vpfill_range[0], vpfill_range[1])
            
            # The length of the initial straight centerline.
            s_init = 5.0 * n_bends * W 
            patch = math.sqrt((self.Xmax - self.Xmin) ** 2 + 4 * (self.Ymax - self.Ymin) ** 2) - (self.Xmax - self.Xmin)
            
            # Make sure the channel crosses the model.
            if s_init < self.Xmax - self.Xmin + patch:
                s_init = self.Xmax - self.Xmin + patch 
            
            # Sampling interval of the channel centerline.
            if delta_s is None:  
                delta_s = s_init / 1000
            delta_s = max(50, delta_s)
            
            # X-coordinate of the channel centerline where the model starts.
            xmin = s_init * X_pos_range[0]
            xmax = s_init * X_pos_range[1]
            X_pos = random.uniform(xmin, xmax)
            if X_pos < patch / 2:
                X_pos = patch / 2
            elif X_pos > s_init + self.Xmin - self.Xmax - patch / 2:
                X_pos = s_init + self.Xmin - self.Xmax - patch / 2
            else:
                pass 
            
            # Y-coordinate of the channel's starting point.
            ymin = self.Ymin + Y_pos_range[0] * (self.Ymax - self.Ymin)
            ymax = self.Ymin + Y_pos_range[1] * (self.Ymax - self.Ymin)
            Y_pos = random.uniform(ymin, ymax)
            
            # Z-coordinate of the channel.
            if len(horizon_in_range) > 0:
                idx = random.choice(horizon_in_range)
                self.horizon[idx].channel = 1
                Z_pos = self.horizon[idx].z
                horizon_in_range.remove(idx)
            else:
                print("All horizons have been used, actual channel number: %d" % (self.n_meandering - 1))
                break
            
            # Add parameters to table.
            meander_parameter.add_row([n + 1, X_pos, Y_pos, Z_pos, strike, W, D, 
                                       s_init, n_bends, n_iter, dt, kl, Cf, vpfill])
            
            # Generate meandering channels.
            channels, oxbows, _ = make_meandering_channel(W=W, D=D, kl=kl, Cf=Cf, dt=dt, delta_s=delta_s, 
                                                          n_iter=n_iter, n_bends=n_bends, perturb_range=perturb_range, 
                                                          pad_up=pad_up, pad_down=pad_down, save_iter=save_iter, 
                                                          co_offset=co_offset, kv=kv, aggr=aggr, 
                                                          t_incision=t_incision, t_aggradation=t_aggradation,
                                                          Y_pos=Y_pos, Z_pos=Z_pos, 
                                                          s_init=s_init, mute=mute, seed=seed)
            # Store channel outputs.            
            self.channel_out.append({'channel': channels, 
                                     'oxbow': oxbows,
                                     'xpos': X_pos,  
                                     'save_iter': save_iter})
            
            # Number of saved channels during migration.
            n_centerline = len(channels)
            
            # Check requirement for deposition.
            if self.dX != self.dY:
                raise ValueError("Model dX != dY.")
            
            # Create channel deposits centerline by centerline.
            if show_migration:
                i_start = 1
            else:
                i_start = n_centerline - 1
            for i_ctl in range(i_start, n_centerline):  
                
                # Get centerline coordinates.
                X_ctl, Y_ctl, Z_ctl = channels[i_ctl].x, channels[i_ctl].y, channels[i_ctl].z
                
                # Select the centerline segment in target area.
                ind = (X_ctl >= X_pos - patch / 2) & (X_ctl <= X_pos + self.Xmax - self.Xmin + patch / 2)
                X_ctl, Y_ctl, Z_ctl = X_ctl[ind], Y_ctl[ind], Z_ctl[ind]
                
                # Resample centerline according to model's resolution.
                if delta_s > self.dX:
                    X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, self.dX)
                
                # Rotate the channel centerline by its strike.
                # Rotation matrix.
                R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],  
                              [math.sin(math.radians(strike)), math.cos(math.radians(strike))]], 
                             dtype=np.float32)
                
                # Rotation center.
                center_X, center_Y = (self.Xmax - self.Xmin) / 2 + X_pos, Y_pos  
                
                # Rotate channel centerline.
                [X_ctl, Y_ctl] = R @ [X_ctl - center_X, Y_ctl - center_Y]  
                X_ctl += center_X
                Y_ctl += center_Y
                
                # Rasterize channel centerline and compute distance to centerline on X-Y plane.
                dist = compute_centerline_distance(X_ctl, Y_ctl, X_pos, self.Ymin, self.dX,
                                                   self.nX, self.nY)
                
                # Initialize topography.
                topo = np.ones(shape=[self.nX, self.nY], dtype=np.float32) * Z_ctl[0]
                
                # Channel erosion.
                if incision_mode == 'random':
                    mode = random.choice([1, 2])
                else:
                    mode = incision_mode
                ze = erosion_surface(cl_dist=dist, zpos=Z_ctl[0], W=Wx, D=Dx, mode=mode)
                ze[ze < self.Zmin] = self.Zmin  # Limit in model's z-range.
                ze[ze > self.Zmax] = self.Zmax  # Limit in model's z-range.
                
                # Update topography.
                topo = np.maximum(topo, ze)
                
                # Channel fill & Vp.
                if epsilon_range is not None:  # Override Vp fill.
                    vp_upper = self.horizon[idx].vp  # Vp of the upper layer.
                    vp_lower = self.horizon[idx+1].vp  # Vp of the lower layer.
                    epsilon = random.uniform(epsilon_range[0], epsilon_range[1])
                    vpfill_1 = (1 + epsilon) * vp_upper
                    vpfill_2 = (1 - epsilon) * vp_upper
                    if abs(vpfill_1 - vp_lower) > abs(vpfill_2 - vp_lower):
                        vpfill = vpfill_1
                    else:
                        vpfill = vpfill_2
                zl = lag_surface(cl_dist=dist, zpos=Z_ctl[0], h_lag=Dx, D=Dx)
                zl[dist > Wx] = 1e10  # Channel fill deposits inside the channel.
                zl[zl < self.Zmin] = self.Zmin  # Limit in model's Z-range.
                zl[zl > self.Zmax] = self.Zmax  # Limit in model's Z-range.
                index = np.argwhere(zl < topo)
                indx, indy = index[:, 0], index[:, 1]
                indz1 = ((zl[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                for i in range(len(indx)):
                    # Channel marker.
                    if instance_label:
                        self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 100 + self.n_meandering
                    else:
                        self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1                           
                    # Channel fill Vp.
                    self.vp[indy[i], indx[i], indz1[i]:indz2[i] + 1] = vpfill
            
        # Print progress.
        if not mute:
            print(meander_parameter)  
        

    def add_meandering_channel_from_database(self, 
                                             params: list[dict], 
                                             database: dict, 
                                             seed: int = None, 
                                             mute: bool = False, 
                                             instance_label: bool = False, 
                                             replacement: bool = False):
        """
        Add meandering river channel to the model.

        Args:
            params (list of dict): Channel parameters.
            database (dict): Channel database information.
            seed (int): Random seed value.
            mute: (bool): Whether to mute printing. Defaults to False.
            instance_label (bool): Whether to create instance label.
                                   If True, will create instance label for each channel (101, 102, ...).
                                   If False, will create semantic label for all channels (0, 1).
                                   Defaults to False.
            replacement (bool): Whether to sampling channel from database with replacement.
                                If True, the sampling channel can occur more than once.
                                If False, the sampled channel will not be sampled again, but be aware that
                                the options may be exausted.
                                Defaults to False.
        """
        # Print progress.
        if not mute:
            sys.stdout.write("\rSimulating meandering channel...")
            
        # Set random state.
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize channel marker.
        if self.channel is None:
            if instance_label:
                # Instance label.
                # 101: channel 1, 102: channel 2, 103: channel 3, ...
                self.channel = np.zeros(self.Z.shape, dtype=np.int16)
            else:
                # Semantic label.
                # 0: non-channel, 1: channel.
                self.channel = np.zeros(self.Z.shape, dtype=np.uint8)
        
        # Check Vp model.
        if self.vp is None:
            raise ValueError("Vp model not found.")
        
        # Get channel database directory.
        database_dir = database['dir']
        
        # Make sure params is a list.
        if isinstance(params, list) is False:
            params = [params]
        
        # Horizon indices.
        if len(self.horizon) > 1:
            horizon_indices = list(range(len(self.horizon) - 1))
        else:
            raise ValueError("Horizon not found.")
        
        # Get channel parameters.
        for item in params:
            n_channel = item['n_channel']  # Number of channels.
            W_range = item['W_range']  # Channel width range.
            D_range = item['D_range']  # Channel depth range.
            R_range = item['R_range']  # Channel width/depth range.
            z_range = item['z_range']  # Channel z position range.
            incision_mode = item['incision_mode']  # Channel incision mode.
            hfill_range = item['hfill_range']  # Channel fill thickness range.
            epsilon_range = item['epsilon_range']  # Control the difference between channel fill mean Vp and upper layer Vp.
            vpfill_std_range = item['vpfill_std_range']  # Channel fill Vp standard deviation range.
            database_name = item['database']  # Channel database name.
            
            # Get horizons id in z range.
            zmin = self.Zmin + z_range[0] * (self.Zmax - self.Zmin)
            zmax = self.Zmin + z_range[1] * (self.Zmax - self.Zmin)
            horizon_in_range = []
            for id in horizon_indices:
                if zmin <= self.horizon[id].z <= zmax:
                    horizon_in_range.append(id)
            
            for i in range(n_channel):
                # Count channels.
                self.n_meandering += 1
                if self.n_meandering > 99:  # 99 channels max.
                    break
                                
                # Channel width.
                if W_range is None:
                    W = random.uniform(150, 1150)
                else:
                    W = random.uniform(W_range[0], W_range[1])
                
                # Channel depth.
                if D_range is None and R_range is not None:
                    R = random.uniform(R_range[0], R_range[1])
                    D = W/R
                elif D_range is not None and R_range is None:
                    D = random.uniform(D_range[0], D_range[1])
                else:
                    raise ValueError("Please define at least one of D_range and R_range.")
                if D < 5:  # Grid dz is 5 m.
                    D = 5
                            
                # Choose a database.
                my_database = get_database(database_dir=database_dir, database_name=database_name, W=W)  
                
                # Randomly choose a channel from the chosen database.
                my_channel = random.choice(database[my_database])
                if not replacement:
                    database[my_database].remove(my_channel)  # The chosen channel will never be chosen again.
                
                # Get channel centerline distance map.
                path = os.path.join(database_dir, my_database+'/'+my_channel)
                distmap = np.load(path+'/data.npy')
                
                # Determine the z-coordinate of the channel.
                if len(horizon_in_range) > 0:
                    id = random.choice(horizon_in_range)
                    self.horizon[id].channel = 1
                    zpos = self.horizon[id].z
                    horizon_in_range.remove(id)
                    horizon_indices.remove(id)
                else:
                    if not mute:
                        print("All horizons have been used.")
                    break
                               
                # Initialize topography.
                topo = np.ones(shape=[self.nX, self.nY], dtype=np.float32) * zpos
                
                # Channel erosion.
                if incision_mode == 'random':
                    mode = random.choice([1, 2])
                else:
                    mode = incision_mode
                ze = erosion_surface(cl_dist=distmap, zpos=zpos, W=W, D=D, mode=mode)
                ze[ze < self.Zmin] = self.Zmin  # Limit in model's z-range.
                ze[ze > self.Zmax] = self.Zmax  # Limit in model's z-range.
                
                # Update topography.
                topo = np.maximum(topo, ze)
                
                # Channel fill & Vp.
                hfill_sum = 0  # Accumulated channel fill thickness.
                vp_upper = self.horizon[id].vp  # Vp of the upper layer.
                vp_lower = self.horizon[id+1].vp  # Vp of the lower layer.
                epsilon = random.uniform(epsilon_range[0], epsilon_range[1])
                vpfill_mean1 = (1 + epsilon) * vp_upper
                vpfill_mean2 = (1 - epsilon) * vp_upper
                if abs(vpfill_mean1 - vp_lower) > abs(vpfill_mean2 - vp_lower):
                    vpfill_mean = vpfill_mean1
                else:
                    vpfill_mean = vpfill_mean2
                vpfill_std = random.uniform(vpfill_std_range[0], vpfill_std_range[1])  # Channel fill Vp standard deviation.    
                while hfill_sum < D:
                    hfill = random.uniform(hfill_range[0], hfill_range[1])  # Thickness of a layer of channel fill.
                    vpfill = random.gauss(mu=vpfill_mean, sigma=vpfill_std)  # Vp of a layer of channel fill.
                    hfill_sum += hfill
                    hfill_sum = min(D, hfill_sum)  # Channel fill thickness can not exceed channel depth.
                    zl = lag_surface(cl_dist=distmap, zpos=zpos, h_lag=hfill_sum, D=D)
                    zl[distmap > W] = 1e10  # Channel fill deposits inside the channel.
                    zl[zl < self.Zmin] = self.Zmin  # Limit in model's Z-range.
                    zl[zl > self.Zmax] = self.Zmax  # Limit in model's Z-range.
                    index = np.argwhere(zl < topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((zl[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    for i in range(len(indx)):
                        # Channel marker.
                        if instance_label:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 100 + self.n_meandering
                        else:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1                           
                        # Channel fill Vp.
                        arr_shape = self.vp[indy[i], indx[i], indz1[i]:indz2[i] + 1].shape
                        vpfill_noisy = np.random.normal(vpfill, 100, arr_shape)  # Noisy channel fill vp.
                        vpfill_noisy = np.clip(vpfill_noisy, a_min=0, a_max=None)  # Avoid negative values.
                        self.vp[indy[i], indx[i], indz1[i]:indz2[i] + 1] = vpfill_noisy
                    
                    # Update topography.
                    topo = np.minimum(topo, zl)
                    
        # Print progress.
        if not mute:
            sys.stdout.write(" Done.\n")

    
    def add_tributary_channel_from_database(self, 
                                            params: list[dict], 
                                            database: dict,
                                            instance_label: bool = True,
                                            replacement: bool = False,   
                                            seed: int = None,  
                                            mute: bool = False):
        """
        Add tributary channels to the model.

        Args:
        params (list of dict): channel parameters. 
                               Each dictionary in the list defines the parameters
                               of a set of channels. The items in each dictionary are:
                               1. n_channel (int): Number of channels.
                               2. z_range (list of float): Channel's uppermost z-coordinate range.
                                                            In the format of [min, max], where each element
                                                            is a fraction between 0 and 1, with 0 means
                                                            the minima of the model's z-coordinate and 1
                                                            means the maxima of the model's z-coordinate.
                                                            The uppermost z-coordinate of each channel will
                                                            be chosen randomly within this range.
                                3. hfill_range (list of float): Thickness range of an individual channel
                                                                fill layer [m]. 
                                                                In the format of [min, max], the thickness
                                                                of each channel fill layer will be randomly
                                                                chosen within this range.
                                                                The channel fill layers keep
                                                                piling up until exceeding the channel's
                                                                uppermost z-coordinate.
                                4. epsilon_range (list of float): Epsilon value controls the contrast between
                                                                    the Vp of channel fill and the Vp of its 
                                                                    upper layer, with 0 means no difference
                                                                    and 1 means the highest contrast.
                                                                    In the format of [min, max].
                                                                    The eplison value of each channel will be 
                                                                    randomly chosen within this range.
                                5. vp_std_range (list of float): Channel fill layer Vp standard deviation range.
                                                                    In the format of [min, max].
                                                                    The Vp standard deviation of each layer will
                                                                    be randomly chosen within this range.
                                6. vp_fill_range (list of float): Bound channel fill Vp within this range.
                                                                    In the format of [min, max].
                                                    
        database (dict): Channel topography database.
                            Items in this dictionary are:
                            1. dir: Directory of the database.
                            2. id: Channel topography folder names.
                            
        instance_label (bool): Whether to create instance label.
                                If True, will create instance label for each channel (101, 102, ...).
                                If False, will create semantic label for all channels (0, 1).
                                Defaults to False. 
                            
        seed (int, optional): Seed value to generate random numbers. 
                                Defaults to None, which is to use current system time.
        
        mute (bool): Whether to mute printing.
                        Defaults to False.
        """
        # Printing.
        if not mute:
            sys.stdout.write("\rSimulating tributary channel...") 
        
        # Set random state.
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize channel marker.
        if self.channel is None:
            if instance_label:
                # Instance label.
                # 201: channel 1, 202: channel 2, 203: channel 3, ...
                self.channel = np.zeros(self.Z.shape, dtype=np.int16)
            else:
                # Semantic label.
                # 0: non-channel, 1: channel.
                self.channel = np.zeros(self.Z.shape, dtype=np.uint8)
            
        # Check Vp model.
        if self.vp is None:
            raise ValueError("Vp model not found.")
        
        # Get channel database directory.
        database_dir = database['dir']
        
        # Horizon indices.
        if len(self.horizon) > 1:
            horizon_indices = list(range(len(self.horizon) - 1))
        else:
            raise ValueError("Horizon not found.")
        
        # Get channel parameters.
        for item in params:
            n_channel = item['n_channel']  # Number of channels.
            z_range = item['z_range']  # Channel z position range.
            hfill_range = item['hfill_range']  # Channel fill thickness range.
            epsilon_range = item['epsilon_range']  # Control the difference between channel fill mean Vp and upper layer Vp.
            vpfill_std_range = item['vpfill_std_range']  # Channel fill Vp standard deviation range.
            
            # Get horizons id in z range.
            zmin = self.Zmin + z_range[0] * (self.Zmax - self.Zmin)
            zmax = self.Zmin + z_range[1] * (self.Zmax - self.Zmin)
            horizon_in_range = []
            for id in horizon_indices:
                if zmin <= self.horizon[id].z <= zmax:
                    horizon_in_range.append(id)
            
            for i in range(n_channel):
                # Count channels.
                self.n_tributary += 1
                if self.n_tributary > 99:  # Maximum 99 channels.
                    break
                                       
                # Randomly choose a tributary channel topography from the database.
                my_topo = random.choice(database['id'])
                if not replacement:
                    database['id'].remove(my_topo)  # The chosen topography will never be chosen again.
                fp = os.path.join(database_dir, my_topo+'/data.npy')
                topo = np.load(fp)  # Channel topography.
                topo[topo < self.dZ] = 0
                            
                # Determine the z-coordinate of the channel.
                if len(horizon_in_range) > 0:
                    id = random.choice(horizon_in_range)
                    self.horizon[id].channel = 2
                    zpos = self.horizon[id].z
                    horizon_in_range.remove(id)
                    horizon_indices.remove(id)
                else:
                    if not mute:
                        print("All horizons have been used.")
                    break
                                
                # Place the channel topography at the selected layer boundary.
                topo_erosion = topo + zpos
                
                # Topography before channel erosion.
                topo_init = np.ones(shape=[self.nX, self.nY], dtype=np.float32) * zpos
                
                # Initial topography of deposition.
                topo_depo = topo_erosion.copy()
            
                # Channel fill & Vp.
                vp_upper = self.horizon[id].vp  # Vp of the upper layer.
                vp_lower = self.horizon[id+1].vp  # Vp of the lower layer.
                epsilon = random.uniform(epsilon_range[0], epsilon_range[1])
                vpfill_mean1 = (1 + epsilon) * vp_upper
                vpfill_mean2 = (1 - epsilon) * vp_upper
                if abs(vpfill_mean1 - vp_lower) > abs(vpfill_mean2 - vp_lower):
                    vpfill_mean = vpfill_mean1
                else:
                    vpfill_mean = vpfill_mean2
                vpfill_std = random.uniform(vpfill_std_range[0], vpfill_std_range[1])  # Channel fill Vp standard deviation.    
                
                topo = topo_erosion.copy()  # Current topography.
                
                while (topo > topo_init).any():
                    hfill = random.uniform(hfill_range[0], hfill_range[1])  # Thickness of a channel fill layer.
                    vpfill = random.gauss(mu=vpfill_mean, sigma=vpfill_std)  # Vp of a channel fill layer.
                    topo_depo -= hfill  # Update deposition topography.
                    topo_depo = np.maximum(topo_init, topo_depo)  # Deposition boundary is the initial topography.
                    index = np.argwhere(topo_depo < topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((topo_depo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    for i in range(len(indx)):
                        # Channel marker.
                        if instance_label:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 200 + self.n_tributary
                        else:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i] + 1] = 1
                        # Channel fill Vp.
                        arr_shape = self.vp[indy[i], indx[i], indz1[i]:indz2[i] + 1].shape
                        vpfill_noisy = np.random.normal(vpfill, 100, arr_shape)  # Noisy channel fill vp.
                        vpfill_noisy = np.clip(vpfill_noisy, a_min=0, a_max=None)  # Avoid negative values. 
                        self.vp[indy[i], indx[i], indz1[i]:indz2[i] + 1] = vpfill_noisy
                    
                    topo = topo_depo.copy()  # Update current topography.
        
        # Printing.
        if not mute:
            sys.stdout.write(" Done.\n")
                        
                                  
    def add_submarine_channel(self, 
                              N: int = 1, 
                              X_pos_range: list = [0.0, 1.0], 
                              Y_pos_range: list = [0.2, 0.8], 
                              Z_pos_range: list = [0.2, 0.8], 
                              strike_range: list[float] = [0, 360],
                              W_range: list[float] = [100, 200], 
                              D_range: list[float] = [20, 30], 
                              kl_range: list[float] = [40, 50], 
                              Cf_range: list[float] = [0.03, 0.05], 
                              co_offset: int = 20,  
                              h_oxbow_range: list[float] = [2, 5], 
                              h_levee_range: list[float] = [1, 2], 
                              w_levee_range: list[float] = [5e3, 1e4], 
                              vp_oxbow_range: list[float] = [4000, 5000],  
                              vp_pointbar_range: list[float] = [3000, 4000], 
                              vp_levee_range: list[float] = [5000, 6500],
                              vp_cap_std: float = 300,  
                              n_iter_range: list[int] = [1000, 1500], 
                              dt: float = 0.1, 
                              save_iter: int = 10,
                              t_inci: list[list] = None, 
                              t_aggr: list[list] = None,
                              n_inci_range: list[int] = [2, 3], 
                              n_aggr_range: list[int] = [2, 3],   
                              kvi_range: list[float] = [5, 10], 
                              kva_range: list[float] = [5, 10],
                              dt_vertical: float = 0.1, 
                              rt_vertical: float = 0.5, 
                              delta_s: float = None, 
                              n_bends_range: list[int] = [40, 50], 
                              perturb_range: list[float] = [0.01, 0.03], 
                              pad_up: int = 1, 
                              pad_down: int = 0,
                              instance_label: bool = True,  
                              seed: int = None, 
                              mute: bool = False):
        """
        Simulate submarine channels.
        
        :param N: (Integer) - The number of channels.
        :param X_pos_range: (List of floats) - [min, max]. Range of channel centerline's X-coordinate which the model starts at (Unit: meter).
                                               Default range is [0, s_init (straight centerline's length) - X_range).
        :param Y_pos_range: (List of floats) - [min, max]. Range of the initial straight centerline's Y-coordinate (Unit: meter).
                                               Default range is [Y_min + 0.1*Y_range, Y_max - 0.1*Y_range).
        :param Z_pos_range: (List of floats) - [min, max]. Range of the initial straight centerline's Z-coordinate (Unit: meter).
                                               Default range is [Z_min + 0.1*Z_range, Z_max - 0.1*Z_range).
        :param strike_range: (List of floats) - [min, max]. Range of the channel's strike angle (Unit: degree, X-direction as north).
                                                Default range is [0, 360).
        :param W_range: (List of floats) - [min, max]. Range of the channel's width (Unit: meter), assuming uniform width.
                                           Default range is [50, 1500).
        :param D_range: (List of floats) - [min, max]. Range of the channel's depth (Unit: meter). Default range is [20, 200).
        :param kl_range: (List of floats) - [min, max]. Range of the migration rate constant (m/year).
                                            Default range is [10, 50).
        :param Cf_range: (List of floats) - [min, max]. Range of the friction factor. Default range is [0.05, 0.1).
        :param co_offset: (Integer) - Number of points from the cutoff points to the centerline points that will be connected.
                                      Default is 20.
        :param h_oxbow_range: (List of floats) - [min, max]. Thickness range of the layers in the oxbow lake (Unit: meter).
                                                 Default range is [Z_resolution, channel depth).
        :param vp_oxbow_range: (List of floats) - [min, max]. Vp range of the oxbow lake fill layers (Unit: m/s).
                                                  Default range is [3000, 6500).
        :param delta_s: (Floats) - Sampling interval along the channel's centerline.
                                   Default is self-adaptive according to the length of initial straight centerline (s_init // 600).
        :param n_bends_range: (List of integers) - [min, max]. Range of the number of bends in the initial centerline.
                                                   Default range is [10, 20).
        :param perturb_range: (List of floats) - [min, max]. Range of perturbation amplitude for centerline initialization (Unit: meter).
                                                 Default range is [200, 500).
        :param nit_range: (List of integers) - [min, max]. Range of the number of iteration. Default range is [500, 2000).
        :param dt: (Float) - Time interval of the migration (year). Default is 0.1.
        :param save_it: (Integer) - Save centerline for every "save_it" iteration. Default is 10.
        :param pad_up: (Integer) - Number of padding points at upstream to fix the centerline. Default is 5.
        :param pad_down: (Integer) - Number of padding points at downstream to fix the centerline. Default is 0.
        :param seed: (Integer) - The seed value needed to generate a random number. 
                                 Default value is None, which is to use the current system time.
        :param mute: (Bool) - If True, will not print anything. Default is False.
        """
        # Set random seed value.
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize outputs.
        if self.channel_out is None:
            self.channel_out = []
        self.topo_out = []
        
        # Check Vp model.
        if self.vp is None:
            raise ValueError("Vp model not found.")
        
        # Store original Vp model.
        vp_origin = self.vp.copy()
        
        # Initialize channel markers.
        if self.channel is None:
            if instance_label:
                # Instance label.
                # 301: channel 1, 302: channel 2, 303: channel 3, ...
                self.channel = np.zeros(self.Z.shape, dtype=np.int16)
            else:
                # Semantic label.
                # 0: non-channel, 1: channel.
                self.channel = np.zeros(self.Z.shape, dtype=np.uint8)
            
        # Initialize facies model.
        if self.facies is None:
            self.facies = np.zeros(self.Z.shape, dtype=np.uint8)

        # Initialize table.
        meander_parameter = PrettyTable()
        meander_parameter.field_names = ['Number', 
                                         'X (m)', 
                                         'Y (m)', 
                                         'Z (m)', 
                                         'strike (degree)', 
                                         'width (m)', 
                                         'depth (m)', 
                                         's_init (m)',
                                         'n_bends', 
                                         'iteration', 
                                         'dt (year)', 
                                         'migration rate constant (m/year)',
                                         'friction factor', 
                                         'h_levee (m)', 
                                         'w_levee (m)',
                                         'Vp pointbar (m/s)', 
                                         'Vp levee (m/s)', 
                                         'Vp oxbow (m/s)']
        meander_parameter.float_format = '.2'
        
        # Horizon indices.
        if len(self.horizon) <= 1:
            raise ValueError("Horizon not found.")
        else:
            horizon_indices = list(range(len(self.horizon) - 1))
        
        # Get horizon id in z range.
        zmin = self.Zmin + Z_pos_range[0] * (self.Zmax - self.Zmin)
        zmax = self.Zmin + Z_pos_range[1] * (self.Zmax - self.Zmin)
        horizon_in_range = []
        for i in horizon_indices:
            if zmin <= self.horizon[i].z <= zmax:
                horizon_in_range.append(i)
        
        # Simulation begins.
        for n in range(N):
            # Count channels.
            self.n_submarine += 1
            if self.n_submarine > 99:  # Maximum 99 channels.
                break
        
            # Print progress.
            if not mute:
                print("Simulating submarine channel [%d/%d]" % ((n+1), N))
                
            # Assign parameters.
            strike = random.uniform(strike_range[0], strike_range[1])
            n_bends = random.randint(n_bends_range[0], n_bends_range[1])
            n_iter = random.randint(n_iter_range[0], n_iter_range[1])
            W = random.uniform(W_range[0], W_range[1])
            D = random.uniform(D_range[0], D_range[1])
            h_levee = random.uniform(h_levee_range[0], h_levee_range[1])
            w_levee = random.uniform(w_levee_range[0], w_levee_range[1])
            kl = random.uniform(kl_range[0], kl_range[1])
            kvi = random.uniform(kvi_range[0], kvi_range[1])
            if kva_range is None:
                kva = kvi
            else:
                kva = random.uniform(kva_range[0], kva_range[1])
            Cf = random.uniform(Cf_range[0], Cf_range[1])
            
            # Generate vertical trajectory.
            if t_inci is None and t_aggr is None:
                n_inci = random.randint(n_inci_range[0], n_inci_range[1])
                n_aggr = random.randint(n_aggr_range[0], n_aggr_range[1])
                t_inci, t_aggr = channel_vertical_trajectory(dt=dt_vertical, 
                                                             rt=rt_vertical, 
                                                             n_inci=n_inci, 
                                                             n_aggr=n_aggr)
            
            # The length of the initial straight centerline.
            s_init = 5.0 * n_bends * W 
            patch = math.sqrt((self.Xmax - self.Xmin) ** 2 + \
                              4 * (self.Ymax - self.Ymin) ** 2) - \
                    (self.Xmax - self.Xmin)
            # Make sure the channel crosses the model.
            if s_init < self.Xmax - self.Xmin + patch:
                s_init = self.Xmax - self.Xmin + patch 
            
            # Sampling interval of the channel centerline.
            if delta_s is None:  
                delta_s = s_init // 1000
            delta_s = max(50, delta_s)
            
            # X-coordinate of the channel centerline where the model starts.
            xmin = s_init * X_pos_range[0]
            xmax = s_init * X_pos_range[1]
            X_pos = random.uniform(xmin, xmax)
            if X_pos < patch / 2:
                X_pos = patch / 2
            elif X_pos > s_init + self.Xmin - self.Xmax - patch / 2:
                X_pos = s_init + self.Xmin - self.Xmax - patch / 2
            else:
                pass
            
            # Y-coordinate of the channel's starting point.
            ymin = self.Ymin + Y_pos_range[0] * (self.Ymax - self.Ymin)
            ymax = self.Ymin + Y_pos_range[1] * (self.Ymax - self.Ymin)
            Y_pos = random.uniform(ymin, ymax)
            
            # Z-coordinate of the channel.
            if len(horizon_in_range) > 0:
                idx = random.choice(horizon_in_range)
                self.horizon[idx].channel = 3
                Z_pos = self.horizon[idx].z
                vp_cap = self.horizon[idx].vp
                horizon_in_range.remove(idx)
            else:
                if not mute:
                    print("All horizons have been used.")
                break
            
            # Add parameters to table.
            meander_parameter.add_row([n + 1, 
                                       X_pos, 
                                       Y_pos, 
                                       Z_pos, 
                                       strike, 
                                       W, 
                                       D, 
                                       s_init, 
                                       n_bends, 
                                       n_iter, 
                                       dt, 
                                       kl, 
                                       Cf,
                                       h_levee, 
                                       w_levee, 
                                       vp_pointbar_range, 
                                       vp_levee_range, 
                                       vp_oxbow_range])
            
            # Initialize channel list.
            channels = []
            
            # Initialize oxbow-lake list.
            oxbow_per_channel = []
            
            # Initialize an array to store channel's z-coordinates.
            cz = np.zeros((n_iter, 2), dtype=np.float32)
            
            # Initialize channel centerline.
            X_ctl, Y_ctl, Z_ctl = initialize_centerline_V0(s_init, W, Y_pos, Z_pos, delta_s, n_bends, perturb_range)
            
            # Resample centerline.
            X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
            
            # Store current channel parameters.
            channels.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
            
            # Channel migration.
            for it in range(n_iter):
                
                # Channel incision.
                if t_inci is not None:
                    for ti in t_inci:
                        if (it > ti[0] * n_iter) & (it <= ti[1] * n_iter):
                            Z_ctl += kvi * dt
                
                # Channel aggradation.
                if t_aggr is not None:
                    for ta in t_aggr:
                        if (it > ta[0] * n_iter) & (it <= ta[1] * n_iter):
                            Z_ctl -= kva * dt
                
                # Limit aggradation range.
                Z_ctl[Z_ctl < Z_pos] = Z_pos
                
                # Compute curvelength.
                dx, dy, ds, s = compute_curvelength(X_ctl, Y_ctl)
                
                # Compute curvatures.
                c = compute_curvature(X_ctl, Y_ctl)
                
                # Compute migration rate.
                R1 = compute_migration_rate(curv=c, ds=ds, W=W, kl=kl, Cf=Cf, D=D, pad_up=pad_up, pad_down=pad_down)
                
                # Adjust migration rate.
                R1 = (s[-1] / (X_ctl[-1] - X_ctl[0])) ** (-2 / 3.0) * R1
                
                # Compute new coordinates after migration.
                ns = len(R1)
                dx_ds = dx[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
                dy_ds = dy[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
                X_ctl[pad_up:ns - pad_down] = X_ctl[pad_up:ns - pad_down] + R1[pad_up:ns - pad_down] * dy_ds * dt
                Y_ctl[pad_up:ns - pad_down] = Y_ctl[pad_up:ns - pad_down] - R1[pad_up:ns - pad_down] * dx_ds * dt
                
                # Resample centerline.
                X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
                
                # Find and execute cutoff.
                X_ox, Y_ox, Z_ox, X_ctl, Y_ctl, Z_ctl = execute_cutoff(X_ctl, Y_ctl, Z_ctl, delta_s, W, co_offset)
                
                # Resample centerline.
                X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
                
                # Store oxbow-lake parameters.
                oxbow_per_channel.append(Oxbow(X_ox, Y_ox, Z_ox, W, D))
                
                # Store channel parameters.
                if it > 0 and it % save_iter == 0:
                    channels.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
                
                # Store channel'z z-coordinates.
                cz[it, 0] = it
                cz[it, 1] = Z_ctl[0]    
                
                # Print progress.
                if not mute:
                    sys.stdout.write('\rChannel migration progress:%.2f%%' % ((it + 1) / n_iter * 100))
                    if it == n_iter - 1:
                        sys.stdout.write("\n")
            
            # Store channel outputs.            
            self.channel_out.append({'channel': channels, 
                                     'oxbow': oxbow_per_channel,
                                     'xpos': X_pos,  
                                     'save_iter': save_iter, 
                                     'zpos': cz})
            
            # Number of saved channels during migration.
            n_centerline = len(channels)
            
            # Check requirement for deposition.
            if self.dX != self.dY:
                raise ValueError("Model dX != dY.")
            
            # Initialize topography.
            topo_set = []
            topo_out = []
            topo = np.ones(shape=[self.nX, self.nY], dtype=np.float32) * Z_pos
            topo_set.append(topo)
            topo_out.append({'data': topo, 
                             'face': 0})
            
            # Initialize global oxbow lake centerline distance with a large number.
            oxbow_dist = np.ones(shape=[self.nX, self.nY], dtype=np.float32) * 1e10
            
            # Create channel deposits centerline by centerline.
            for i_ctl in range(n_centerline):
                
                # Get centerline coordinates.
                X_ctl, Y_ctl, Z_ctl = channels[i_ctl].x, channels[i_ctl].y, channels[i_ctl].z
                
                # Select the centerline segment in target area.
                ind = (X_ctl >= X_pos - patch / 2) & (X_ctl <= X_pos + self.Xmax - self.Xmin + patch / 2)
                X_ctl, Y_ctl, Z_ctl = X_ctl[ind], Y_ctl[ind], Z_ctl[ind]
                
                # Resample centerline according to model's resolution.
                if delta_s > self.dX:
                    X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, self.dX)
                
                # Rotate the channel centerline by its strike.
                # Rotation matrix.
                R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],  
                              [math.sin(math.radians(strike)), math.cos(math.radians(strike))]], 
                             dtype=np.float32)
                # Rotation center.
                center_X, center_Y = (self.Xmax - self.Xmin) / 2 + X_pos, Y_pos  
                # Rotate channel centerline.
                [X_ctl, Y_ctl] = R @ [X_ctl - center_X, Y_ctl - center_Y]  
                X_ctl += center_X
                Y_ctl += center_Y
                
                # Rasterize channel centerline and compute distance to centerline on X-Y plane.
                dist = compute_centerline_distance(X_ctl, Y_ctl, X_pos, self.Ymin, self.dX,
                                                   self.nX, self.nY)
                
                # Make a list of oxbow lake centerline coordinates between ith and (i-1)th migration.
                X_ox, Y_ox, Z_ox = [], [], []
                
                # Check if cutoff happens between ith and (i-1)th migration.
                if i_ctl > 0: 
                    for i_cn in range((i_ctl - 1) * save_iter, i_ctl * save_iter, 1):
                        # Get oxbow centerline.  
                        X_oxbow = oxbow_per_channel[i_cn].x.copy()                       
                        Y_oxbow = oxbow_per_channel[i_cn].y.copy()
                        Z_oxbow = oxbow_per_channel[i_cn].z.copy()
                        
                        if len(X_oxbow) > 0:
                            # Number of oxbows.
                            n_oxbow = len(X_oxbow)
                            
                            for i_oxbow in range(n_oxbow):
                                # Resample centerline.
                                if delta_s > self.dX:
                                    X_oxbow[i_oxbow], Y_oxbow[i_oxbow], Z_oxbow[i_oxbow] = resample_centerline(X_oxbow[i_oxbow], 
                                                                                                               Y_oxbow[i_oxbow], 
                                                                                                               Z_oxbow[i_oxbow], 
                                                                                                               self.dX)
                                
                                # Select oxbow centerline in target area.
                                ind = (X_oxbow[i_oxbow] >= X_pos - patch / 2) & (X_oxbow[i_oxbow] <= X_pos + self.Xmax - self.Xmin + patch / 2)
                                X_oxbow[i_oxbow], Y_oxbow[i_oxbow], Z_oxbow[i_oxbow] = X_oxbow[i_oxbow][ind], Y_oxbow[i_oxbow][ind], Z_oxbow[i_oxbow][ind]
                                
                                # Rotate oxbow lake centerline by channel strike.
                                if len(X_oxbow[i_oxbow]) > 0:
                                    # Rotation matrix.
                                    R = np.array([[math.cos(math.radians(strike)), -math.sin(math.radians(strike))],
                                                  [math.sin(math.radians(strike)), math.cos(math.radians(strike))]],
                                                 dtype=np.float32)  
                                    
                                    # Rotation center.
                                    center_X, center_Y = (self.Xmax - self.Xmin) / 2 + X_pos, Y_pos
                                    
                                    # Rotate.
                                    [X_oxbow[i_oxbow], Y_oxbow[i_oxbow]] = R @ [X_oxbow[i_oxbow] - center_X, Y_oxbow[i_oxbow] - center_Y]
                                    X_oxbow[i_oxbow] += center_X
                                    Y_oxbow[i_oxbow] += center_Y
                                    
                                    # Assemble the oxbows' coordinates between ith and (i-1)th migration.
                                    X_ox.append(X_oxbow[i_oxbow])
                                    Y_ox.append(Y_oxbow[i_oxbow])
                                    Z_ox.append(Z_oxbow[i_oxbow])
                
                # If cutoffs occur before, compute distance from their centerlines.
                if len(X_ox) > 0:
                    for i_ox in range(len(X_ox)):
                        # Compute distance from oxbow lake centerline.
                        ox_dist = compute_centerline_distance(X_ox[i_ox], 
                                                              Y_ox[i_ox], 
                                                              X_pos, 
                                                              self.Ymin, 
                                                              self.dX, 
                                                              self.nX, 
                                                              self.nY)
                        
                        # Update global oxbow lake centerline distance.
                        oxbow_dist = np.minimum(oxbow_dist, ox_dist)
                        
                        # Oxbow lake erosion.
                        ze = erosion_surface(cl_dist=ox_dist, zpos=Z_ox[i_ox][0], W=W, D=D)
                        # Limit in model's Z-range.
                        ze[ze < self.Zmin] = self.Zmin  
                        ze[ze > self.Zmax] = self.Zmax
                        # Get model indexes.  
                        index = np.argwhere(ze > topo)
                        indx, indy = index[:, 0], index[:, 1]
                        indz1 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                        indz2 = ((ze[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                        for i in range(len(indx)):
                            # Non-channel.
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 0
                            # Facies: background (0).
                            self.facies[indy[i], indx[i], indz1[i]:indz2[i]+1] = 0
                            # Background Vp.
                            self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_origin[indy[i], indx[i], indz1[i]:indz2[i]+1]
                        # Update topography.
                        topo = np.maximum(topo, ze)
                        topo_set.append(topo)
                        topo_out.append({'data': topo, 
                                         'face': 0})
                        
                        # Oxbow lake fill.
                        h_oxbow_sum = 0
                        while h_oxbow_sum < D:
                            h_oxbow = random.uniform(h_oxbow_range[0], h_oxbow_range[1])
                            vp_oxbow = random.uniform(vp_oxbow_range[0], vp_oxbow_range[1])
                            h_oxbow_sum += h_oxbow
                            # Oxbow-lake channel fill can not exceed channel depth.
                            h_oxbow_sum = min(D, h_oxbow_sum)  
                            zl = lag_surface(cl_dist=ox_dist, zpos=Z_ox[i_ox][0], h_lag=h_oxbow_sum, D=D)
                            # Deposit inside oxbow.
                            zl[ox_dist > W] = 1e10
                            # Limit in model's Z-range.  
                            zl[zl < self.Zmin] = self.Zmin  
                            zl[zl > self.Zmax] = self.Zmax
                            # Get model array indexes.
                            index = np.argwhere(zl < topo)
                            indx, indy = index[:, 0], index[:, 1]
                            indz1 = ((zl[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                            indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                            for i in range(len(indx)):
                                # Channel.
                                if instance_label:
                                    self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 300 + self.n_submarine
                                else:
                                    self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 1
                                # Facies: oxbow (4).
                                self.facies[indy[i], indx[i], indz1[i]:indz2[i]+1] = 4
                                # Oxbow lake Vp.
                                self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_oxbow
                            topo = np.minimum(zl, topo)
                            topo_set.append(topo)
                            topo_out.append({'data': topo, 
                                             'face': 4})
                
                # Channel erosion.
                ze = erosion_surface(cl_dist=dist, zpos=Z_ctl[0], W=W, D=D)
                # Limit in model's Z-range.
                ze[ze < self.Zmin] = self.Zmin  
                ze[ze > self.Zmax] = self.Zmax
                # Get model array indexes.
                index = np.argwhere(ze > topo)
                indx, indy = index[:, 0], index[:, 1]
                indz1 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                indz2 = ((ze[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                for i in range(len(indx)):
                    # Non-channel.
                    self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 0
                    # Facies: background (0).
                    self.facies[indy[i], indx[i], indz1[i]:indz2[i]+1] = 0
                    # Background Vp.
                    self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_origin[indy[i], indx[i], indz1[i]:indz2[i]+1]
                topo = np.maximum(topo, ze)
                topo_set.append(topo)
                topo_out.append({'data': topo, 
                                 'face': 0})
                
                # Before the latest channel.
                if i_ctl != n_centerline - 1:
                    # Pointbar.    
                    vp_pointbar = random.uniform(vp_pointbar_range[0], vp_pointbar_range[1])
                    zpb = pointbar_surface(cl_dist=dist, z=Z_ctl, W=W, D=D)
                    # zpb[oxbow_dist <= W] = 1e10  # Clear pointbar deposits inside oxbow lake.
                    # Limit in model's Z-range.
                    zpb[zpb < self.Zmin] = self.Zmin  
                    zpb[zpb > self.Zmax] = self.Zmax
                    # Get model array indexes.
                    index = np.argwhere(zpb < topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((zpb[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    for i in range(len(indx)):
                        # Channel.
                        if instance_label:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 300 + self.n_submarine
                        else:
                            self.channel[indy[i], indx[i], indz1[i]:indz2[i]+1] = 1
                        # Facies: point-bar (2).
                        self.facies[indy[i], indx[i], indz1[i]:indz2[i]+1] = 2
                        # Pointbar Vp.
                        self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_pointbar
                    topo = np.minimum(topo, zpb)
                    topo_set.append(topo)
                    topo_out.append({'data': topo, 
                                     'face': 2})
                    
                    # Natural levee.
                    vp_levee = random.uniform(vp_levee_range[0], vp_levee_range[1])
                    zlv = levee_surface(cl_dist=dist, h_levee=h_levee, w_levee=w_levee, W=W, tp=topo)
                    # Limit in model's Z-range.
                    zlv[zlv < self.Zmin] = self.Zmin  
                    zlv[zlv > self.Zmax] = self.Zmax
                    # Get model array indexes.
                    index = np.argwhere(zlv < topo)
                    indx, indy = index[:, 0], index[:, 1]
                    indz1 = ((zlv[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    indz2 = ((topo[indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
                    # Inner levee.
                    zlv_inner = np.maximum(topo_set[0], zlv)  
                    index_inner = np.argwhere(zlv_inner < topo)
                    indx_inner, indy_inner = index_inner[:, 0], index_inner[:, 1]
                    indz1_inner = ((zlv_inner[indx_inner, indy_inner] - self.Zmin) / self.dZ).astype(np.int32)
                    indz2_inner = ((topo[indx_inner, indy_inner] - self.Zmin) / self.dZ).astype(np.int32)
                    for i in range(len(indx)):
                        # Natural levee Vp.
                        self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_levee
                        # Facies: natural levee (3)
                        self.facies[indy[i], indx[i], indz1[i]:indz2[i]+1] = 3
                    for i in range(len(indx_inner)):
                        # Channel fill label.
                        if instance_label:
                            self.channel[indy_inner[i], indx_inner[i], indz1_inner[i]:indz2_inner[i]+1] = 300 + self.n_submarine
                        else:
                            self.channel[indy_inner[i], indx_inner[i], indz1_inner[i]:indz2_inner[i]+1] = 1
                    topo = np.minimum(topo, zlv)
                    topo_set.append(topo)
                    topo_out.append({'data': topo, 
                                     'face': 3})
                
                # Print progress.
                if not mute:
                    sys.stdout.write('\rChannel deposition progress:%.2f%%  centerline[%d/%d]' %
                                    ((i_ctl + 1) / n_centerline * 100, i_ctl + 1, n_centerline))
                    if i_ctl == n_centerline - 1:
                        sys.stdout.write("\n")
            
            # Modify upper layer's velocity.
            index = np.argwhere(topo_set[-1] > topo_set[0])
            indx, indy = index[:, 0], index[:, 1]
            indz1 = ((topo_set[0][indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
            indz2 = ((topo_set[-1][indx, indy] - self.Zmin) / self.dZ).astype(np.int32)
            for i in range(len(indx)):
                shape = self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1].shape
                vp_noisy = np.random.normal(vp_cap, vp_cap_std, shape)
                self.vp[indy[i], indx[i], indz1[i]:indz2[i]+1] = vp_noisy
            
            # Store topography outputs.
            self.topo_out.append(topo_out)
            
            # Print progress.
            if not mute:
                print(meander_parameter)
        
    
    def resample_z(self, 
                   dz: float = None, 
                   z_range: tuple = None, 
                   param: str|list[str] = 'all', 
                   method: str = 'linear', 
                   mute: bool = False):
        """
        Resample model parameters in z-direction.

        Args:
        dz (float): Sampling spacing [m]. 
                    Defaults to None, 
                    which is to use the model's sampling spacing.
        z_range (tuple): Resampling z range [m]. 
                         Defaults to None, 
                         which is the model's z range.             
        param (str or list of str): Model parameters to be resampled. 
                                    Defaults to 'all', 
                                    which is to resample all model parameters.
                                    Can also be a list of parameter names.
                                    Available model parameters are:
                                    1. vp
                                    2. channel
                                    3. rgt 
        method (str): Resampling method. 
                      Defaults to 'linear'.
                      For other methods please refer to scipy.interpolate.interp1d
                      (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
        mute (bool): Whether to mute printing. 
                     Defaults to False.
        """
        if dz is None:
            dz = self.dZ
        
        if z_range is None:
            z_range = (self.Zmin, self.Zmax)
        
        zmin, zmax = z_range
        ny, nx, _ = self.X.shape
        z_new = np.arange(zmin, zmax, dz, dtype=np.float32)
        
        if not mute:
            print('New Z range: %.2fm-%.2fm [%.2fm]' % 
                  (zmin, zmax, dz))
            print("New model shape (XYZ): [%d x %d x %d]" %
                  (self.Z.shape[1], self.Z.shape[0], len(z_new)))
        
        if 'vp' in param or param == 'all':
            vp_new = np.zeros((ny, nx, len(z_new)), dtype=self.vp.dtype)
            for i in range(ny):
                for j in range(nx):
                    if not mute:
                        sys.stdout.write('\rResampling Vp: %.2f%%' % 
                                         ((i*nx+j+1) / (ny*nx) * 100))
                    f = interpolate.interp1d(self.Z[i, j, :], 
                                             self.vp[i, j, :], 
                                             kind=method, 
                                             fill_value=(self.vp[i, j, 0], self.vp[i, j, -1]), 
                                             assume_sorted=True, 
                                             bounds_error=False)
                    vp_new[i, j, :] = f(z_new)
            self.vp = vp_new.copy()
            if not mute:
                sys.stdout.write("\n")
        
        if 'rgt' in param or param == 'all':
            rgt_new = np.zeros((ny, nx, len(z_new)), dtype=self.rgt.dtype)
            for i in range(ny):
                for j in range(nx):
                    if not mute:
                        sys.stdout.write("\rResampling RGT: %.2f%%" % 
                                         ((i*nx+j+1) / (ny*nx) * 100))
                    f = interpolate.interp1d(self.Z[i, j, :], 
                                             self.rgt[i, j, :], 
                                             kind=method, 
                                             fill_value='extrapolate', 
                                             assume_sorted=True)
                    rgt_new[i, j, :] = f(z_new)
            self.rgt = rgt_new.copy()
            if not mute:
                sys.stdout.write("\n")
        
        if 'channel' in param or param == 'all':
            if self.channel.dtype == np.uint8 or self.channel.dtype == np.int16:
                method_c = 'nearest'
            else:
                method_c = method
            channel_new = np.zeros((ny, nx, len(z_new)), dtype=self.channel.dtype)
            for i in range(ny):
                for j in range(nx):
                    if not mute:
                        sys.stdout.write('\rResampling channel marker: %.2f%%' % ((i*nx+j+1) / (ny*nx) * 100))
                    f = interpolate.interp1d(self.Z[i, j, :], 
                                             self.channel[i, j, :], 
                                             kind=method_c, 
                                             fill_value=(0, 0), 
                                             assume_sorted=True, 
                                             bounds_error=False)
                    channel_new[i, j, :] = f(z_new)
            self.channel = channel_new.copy()
            if not mute:
                sys.stdout.write("\n")
                
        if 'facies' in param or param == 'all':
            if self.facies.dtype == np.uint8:
                method_c = 'nearest'
            else:
                method_c = method
            facies_new = np.zeros((ny, nx, len(z_new)), dtype=self.facies.dtype)
            for i in range(ny):
                for j in range(nx):
                    if not mute:
                        sys.stdout.write('\rResampling facies: %.2f%%' % ((i*nx+j+1) / (ny*nx) * 100))
                    f = interpolate.interp1d(self.Z[i, j, :], 
                                             self.facies[i, j, :], 
                                             kind=method_c, 
                                             fill_value=(0, 0), 
                                             assume_sorted=True, 
                                             bounds_error=False)
                    facies_new[i, j, :] = f(z_new)
            self.facies = facies_new.copy()
            if not mute:
                sys.stdout.write("\n")
    
    
    def depth2time(self, dz=None, dt=2, nt=256, vp=None, param=None, mute=False):
        # Check Vp model.
        if self.vp is None:
            raise ValueError('No valid Vp model.')
        # Default dz.
        if dz is None:
            dz = self.dZ
        # Initialization.
        if vp is None:
            vp = self.vp.copy()
        else:
            vp = np.ones(self.vp.shape, dtype=np.float32) * vp
        t = np.arange(0, nt*dt, dt)
        self.twt = np.zeros((self.vp.shape[0], self.vp.shape[1], len(t)), dtype=np.float32)
        if param == 'all' or 'vp' in param:
            vp_new = np.zeros((self.vp.shape[0], self.vp.shape[1], len(t)), dtype=np.float32)
        if param == 'all' or 'channel' in param:
            self.channel = self.channel.astype(np.float32)
            channel_new = np.zeros((self.vp.shape[0], self.vp.shape[1], len(t)), dtype=np.int32)
        if param == 'all' or 'ip' in param:
            ip_new = np.zeros((self.vp.shape[0], self.vp.shape[1], len(t)), dtype=np.float32)
        # Depth-time relationship.
        d2t = np.zeros(self.vp.shape, dtype=np.float32)
        for i in range(self.vp.shape[0]):
            for j in range(self.vp.shape[1]):
                if not mute:
                    sys.stdout.write('\rDepth to time conversion: %.2f%%' %
                                    ((i*self.vp.shape[1]+j+1) / (self.vp.shape[0]*self.vp.shape[1]) * 100))
                d2t[i, j, 1:] = np.cumsum(dz / vp[i, j, 1:]) * 1000  # One-way time (ms).
                d2t[i, j, 1:] *= 2  # Two-way time (ms).
                # Depth to time conversion.
                if param == 'all' or 'vp' in param:
                    vp_new[i, j, :] = np.interp(x=t, xp=d2t[i, j, :], fp=self.vp[i, j, :])
                # Depth to time conversion.
                if param == 'all' or 'ip' in param:
                    ip_new[i, j, :] = np.interp(x=t, xp=d2t[i, j, :], fp=self.Ip[i, j, :])
                if param == 'all' or 'channel' in param:
                    f = interpolate.interp1d(d2t[i, j, :], self.channel[i, j, :], kind='nearest', 
                                             fill_value='extrapolate', assume_sorted=True)
                    channel_new[i, j, :] = f(t)
                self.twt[i, j, :] = t
        if not mute:
            sys.stdout.write('\n')
        if param == 'all' or 'vp' in param:
            self.vp = vp_new.copy()
        if param == 'all' or 'channel' in param:
            self.channel = channel_new.copy()
        if param == 'all' or 'ip' in param:
            self.Ip = ip_new.copy()
        if not mute:
            print('Depth to time conversion finished.')
            print('Model extent:')
            print('X range: %.2fm-%.2fm' % (self.Xmin, self.Xmax))
            print('Y range: %.2fm-%.2fm' % (self.Ymin, self.Ymax))
            print('T range: %dms-%dms' % (self.twt.min(), self.twt.max()))
            print('Model resolution (XYT): [%.2fm x %.2fm x %dms]' %
                (self.dX, self.dY, dt))
            print('Model points (XYT): [%d x %d x %d]' % (self.nX, self.nY, self.twt.shape[-1])) 


def extract_isosurface(volume: np.ndarray, 
                       value: float):
    # Get volume shape.
    ny, nx, _ = volume.shape
    
    # Initialize isosurface index.
    si = np.zeros((nx, ny), dtype=np.float32)
    
    # Extract isosurface.
    for i in range(ny):
        for j in range(nx):
            si[i, j] = np.argmin(np.abs(volume[i, j, :] - value))
    
    return si


def compute_f_parallel(i, x=None, y=None, interpolator=None):
    """
    Compute the curved fault plane's z coordinates using the bi-harmonic spline interpolation in parallel .
    :param i: (Integer) - The slice index number (axis=0) of the model's x and y coordinate arrays in fault plane
              coordinate system.
    :param x: (numpy.3darray) - The model's x coordinate array in fault plane coordinate system.
    :param y: (numpy.3darray) - The model's y coordinate array in fault plane coordinate system.
    :param interpolator: (class BiharmonicSpline3D) - The bi-harmonic spline interpolator initialized by
                         random perturbation points near the planar fault plane.
    :return: (List of numpy.2darrays) - A slice (axis=0) of curved fault plane's z coordinates in fault plane
             coordinate system.
    """
    out = interpolator(x[i, :, :], y[i, :, :])
    return out


def compute_Dz_parallel(i, x=None, y=None, f=None, interpolator=None):
    """
    Compute the model's displacement in the fault plane's normal direction.
    :param i: (Integer) - The slice index number (axis=0) of the model's x and y coordinate arrays in fault plane
              coordinate system.
    :param x: (numpy.3darray) - The model's x coordinate array in fault plane coordinate system.
    :param y: (numpy.3darray) - The model's y coordinate array in fault plane coordinate system.
    :param f: (numpy.3darray) - The fault plane's z-coordinate array in fault plane coordinate system.
    :param interpolator: (class BiharmonicSpline3D) - The bi-harmonic spline interpolator initialized by
                         random perturbation points near the planar fault plane.
    :return: (List of numpy.2darrays) - A slice (axis=0) of the model's displacement in the fault plane's
             normal direction.
    """
    out = interpolator(x[i, :, :], y[i, :, :]) - f[i, :, :]
    return out


class Channel:
    """
    Store the channel centerline coordinates, channel width, maximum depth and saving interval.
    """

    def __init__(self, x, y, z, W, D):
        """
        :param x: (numpy.1darray) - x-coordinates of centerline.
        :param y: (numpy.1darray) - y-coordinates of centerline.
        :param z: (numpy.1darray) - z-coordinates of centerline.
        :param W: (Float) - River width.
        :param D: (Float) - River maximum depth.
        """
        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D


class Oxbow:
    """
    Store the oxbow-lake centerline coordinates, oxbow-lake width and maximum depth.
    """

    def __init__(self, xc, yc, zc, W, D):
        """
        :param xc: (numpy.1darray) - x-coordinates of oxbow centerline.
        :param yc: (numpy.1darray) - y-coordinates of oxbow centerline.
        :param zc: (numpy.1darray) - z-coordinates of oxbow centerline.
        :param W: (Float) - Oxbow-lake width.
        :param D: (Float) - Oxbow-lake maximum depth.
        """
        self.x = xc
        self.y = yc
        self.z = zc
        self.W = W
        self.D = D


class Horizon:
    def __init__(self, 
                 z: float, 
                 vp: float, 
                 rgt: float, 
                 channel: int):
        """
        Horizon (the bottom surface of a layer).

        Args:
            z (float): Z-coordinate of the undeformed horizon.
            vp (float): Seismic P-wave velocity of the layer.
            rgt (float): Relative geologic time.
            channel (bool): Whether the horizon has channel erosion. 
        """
        self.z = z
        self.vp = vp
        self.rgt = rgt
        self.channel = channel


def make_meandering_channel(W=200.0, D=20.0, kl=50.0, Cf=0.05, dt=0.1, delta_s=None, n_iter=2000, n_bends=20, 
                            perturb_range=[0.01, 0.02], pad_up=0, pad_down=0, save_iter=20, co_offset=20,
                            kv=3.0, aggr=4.0, t_incision=None, t_aggradation=None, s_init=None,
                            Y_pos=None, Z_pos=None, mute=False, seed=None):
    """
    Make a meandering river channel.

    Args:
        W (Float): Channel width (m). Defaults to 200 m.
        D (Float): Channel depth (m). Defaults to 20 m.
        kl (Float): Migration rate constant (m/year). Defaults to 50 m/y.
        Cf (Float): Chezy's friction factor. Defaults to 0.05.
        dt (Float): Time step (year). Defaults to 0.1 y.
        delta_s (Float): Channel centerline sampling spacing (m).
                         Minimum 50 m. 
                         Defaults to None, which is s_init // 1000.
        n_iter (Int): Number of iteration. Defaults to 2000.
        n_bends (Int): Number of initial channel bends. Defaults to 20.
        perturb_range (List of floats): Perturbation amplitude range for making channel bends, in the format of [min, max].
                                        The actual values will be [min * W, max * W]. 
                                        Defaults to [0.01, 0.02].
        pad_up (Integer): Number of points at channel upstream that will not migrate. Defaults to 0.
        pad_down (Integer): Number of points at channel downstream that will not migrate. Defaults to 0.
        save_iter (Integer): Save channel for every 'save_iter' iteration. Defaults to 20.
        co_offset (Integer): Number of points from the cutoff to the channel centerline. 
                             High value creates straighter channel.
                             Defaults to 20.
        kv (float): incision rate (m/year).
        aggr (float): aggradation rate (m/year).
        t_incision (list): Time steps of incision.
        t_aggradation (list): Time steps of aggradation.
        s_init (float): Initial length of the centerline. If None, will automatically compute a value.
        Y_pos (float): Y-coordinate of the initial centerline's first point. If None, will set to zero.
        Z_pos (float): Z-coordinate of the initial centerline. If None, will set to zero.
        mute (Bool): Whether to mute printing. Defaults to False.
        seed (Int): Random seed value. Defaults to None, which is to use the current system time.

    Returns:
        channels (List of objects): A list of all saved channels during channel migration.
                                    Each channel object has the following attributes:
                                    1. x (Numpy.ndarray): X-coordinates of the channel centerline.
                                    2. y (Numpy.ndarray): Y-coordinates of the channel centerline.
                                    3. z (Numpy.ndarray): Z-coordinates of the channel centerline.
                                    4. W (Float): Channel width.
                                    5: D (Float): Channel depth.
        oxbows (List of objects): A list of all oxbow lakes (abandoned channel) during channel migration.
                                  Each oxbow lake object has the following attributes:
                                  1. x (Numpy.ndarray): X-coordinates of the channel centerline.
                                  2. y (Numpy.ndarray): Y-coordinates of the channel centerline.
                                  3. z (Numpy.ndarray): Z-coordinates of the channel centerline.
                                  4. W (Float): Channel width.
                                  5: D (Float): Channel depth.
        params (Dictionary): Miscellaneous modeling parameters.
    """
    # Random state.
    np.random.seed(seed)
    random.seed(seed)
    
    # Initial channel length.
    if s_init is None: 
        s_init = 5 * W * n_bends
    
    # Sampling interval of the channel centerline.
    if delta_s is None:
        delta_s = s_init / 1000
    delta_s = max(50, delta_s)
    
    # Parameters.
    params = dict(W=W, D=D, kl=kl, Cf=Cf, dt=dt, delta_s=delta_s, n_iter=n_iter, n_bends=n_bends, 
                  perturb_range=perturb_range, pad_up=pad_up, pad_down=pad_down, save_iter=save_iter, 
                  co_offset=co_offset, mute=mute, seed=seed)
    
    # Initialize channel list.
    channels = []
        
    # Initialize oxbow-lake list.
    oxbows = []
    
    # Initialize channel centerline.
    X_ctl, Y_ctl, Z_ctl = initialize_centerline(s_init, W, delta_s, n_bends, perturb_range, 
                                                ypos=Y_pos, zpos=Z_pos)
    
    # Resample centerline so that delta_s is roughly constant.
    X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
    
    # Save current channel parameters.
    channels.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
    
    # Start channel migration.
    for it in range(n_iter):
        # Channel incision.
        if t_incision is not None:
            for t_incise in t_incision:
                if (it > t_incise[0] * n_iter) & (it <= t_incise[1] * n_iter):
                    Z_ctl += kv * dt
        # Channel aggradation.
        if t_aggradation is not None:
            for t_aggr in t_aggradation:
                if (it > t_aggr[0] * n_iter) & (it <= t_aggr[1] * n_iter):
                    Z_ctl -= aggr * dt
        # Compute derivative of x (dx), y (dy), centerline's length (s) and distances between two
        # consecutive points along the centerline.
        dx, dy, ds, s = compute_curvelength(X_ctl, Y_ctl)
        # Compute curvatures at each points on the centerline.
        c = compute_curvature(X_ctl, Y_ctl)
        # Compute migration rate of each point.
        R1 = compute_migration_rate(curv=c, ds=ds, W=W, kl=kl, Cf=Cf, D=D, pad_up=pad_up, pad_down=pad_down)
        # Adjust migration rate.
        R1 = (s[-1] / (X_ctl[-1] - X_ctl[0])) ** (-2 / 3.0) * R1
        # Compute centerline coordinates after migration.
        ns = len(R1)
        dx_ds = dx[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
        dy_ds = dy[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
        X_ctl[pad_up:ns - pad_down] = X_ctl[pad_up:ns - pad_down] + R1[pad_up:ns - pad_down] * dy_ds * dt
        Y_ctl[pad_up:ns - pad_down] = Y_ctl[pad_up:ns - pad_down] - R1[pad_up:ns - pad_down] * dx_ds * dt
        # Resample the centerline so that delta_s is roughly constant.
        X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
        # Find and execute cutoff.
        X_ox, Y_ox, Z_ox, X_ctl, Y_ctl, Z_ctl = execute_cutoff(X_ctl, Y_ctl, Z_ctl, delta_s, W, co_offset)
        # Resample the centerline so that delta_s is roughly constant.
        X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
        # Save oxbow-lake parameters.
        oxbows.append(Oxbow(X_ox, Y_ox, Z_ox, W, D))
        # Save channel parameters.
        if it > 0 and it % save_iter == 0:
            channels.append(Channel(X_ctl, Y_ctl, Z_ctl, W, D))
        if not mute:
            # Print progress.
            sys.stdout.write('\rChannel migration progress:%.2f%%' % ((it + 1) / n_iter * 100))
    if not mute:
        sys.stdout.write('\n')
        
    return channels, oxbows, params


def migrate(X_ctl: np.ndarray, 
            Y_ctl: np.ndarray, 
            Z_ctl: np.ndarray, 
            delta_s: float = 50, 
            W: float = 200, 
            D: float = 10, 
            kl: float = 50, 
            Cf: float = 0.05, 
            pad_up: int = 5, 
            pad_down: int = 5, 
            co_offset: int = 15, 
            dt: float = 0.1, 
            n_iter: float = 1000, 
            mute: bool = False):
    """
    River centerline migration.

    Args:    
    X_ctl (np.ndarray): Centerline's x coordinates [m].
    Y_ctl (np.ndarray): Centerline's y coordinates [m].
    Z_ctl (np.ndarray): Centerline's z coordinates [m].
    delta_s (float): Distance of two consecutive points along centerline [m]. 
                    Defaults to 50 m.
    W (float): River width [m]. 
               Defaults to 200 m.
    D (float): River depth [m]. 
               Defaults to 10 m.
    kl (float): Migration rate constant [m/year]. 
                Defaults to 50 m/year.
    Cf (float): Chezy's friction factor. 
                Defaults to 0.05.
    pad_up (int): Number of fixed points at the beginning of centerline. 
                  Defaults to 5.
    pad_down (int): Number of fixed points at the end of centerline. 
                    Defaults to 5.
    co_offset (int): Distance from cutoff to re-conneted points [points]. 
                     Defaults to 15 points.
    dt (float): Time step [year]. 
                Defaults to 0.1 year.
    n_iter (float): Number of iteration. 
                    Defaults to 1000.
    mute (bool): Whether to mute printing. 
                 Defaults to False.

    Returns:
    X_ctl (np.ndarray): Centerline's x coordiantes after migration.
    Y_ctl (np.ndarray): Centerline's y coordinates after migration.
    Z_ctl (np.ndarray): Centerline's z coordinates after migration.
    """
    
    # Resample centerline.
    X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s=delta_s)
    
    # Start channel migration.
    for it in range(n_iter):
        # Compute derivative of x (dx), y (dy), centerline's length (s) and distances between two
        # consecutive points along the centerline.
        dx, dy, ds, s = compute_curvelength(X_ctl, Y_ctl)
        # Compute curvatures at each points on the centerline.
        c = compute_curvature(X_ctl, Y_ctl)
        # Compute migration rate of each point.
        R1 = compute_migration_rate(curv=c, ds=ds, W=W, kl=kl, Cf=Cf, D=D, pad_up=pad_up, pad_down=pad_down)
        # Adjust migration rate.
        R1 = (s[-1] / (X_ctl[-1] - X_ctl[0])) ** (-2 / 3.0) * R1
        # Compute centerline coordinates after migration.
        ns = len(R1)
        dx_ds = dx[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
        dy_ds = dy[pad_up:ns - pad_down] / ds[pad_up:ns - pad_down]
        X_ctl[pad_up:ns - pad_down] = X_ctl[pad_up:ns - pad_down] + R1[pad_up:ns - pad_down] * dy_ds * dt
        Y_ctl[pad_up:ns - pad_down] = Y_ctl[pad_up:ns - pad_down] - R1[pad_up:ns - pad_down] * dx_ds * dt
        # Resample the centerline so that delta_s is roughly constant.
        X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
        # Find and execute cutoff.
        X_ox, Y_ox, Z_ox, X_ctl, Y_ctl, Z_ctl = execute_cutoff(X_ctl, Y_ctl, Z_ctl, delta_s, W, co_offset)
        # Resample the centerline so that delta_s is roughly constant.
        X_ctl, Y_ctl, Z_ctl = resample_centerline(X_ctl, Y_ctl, Z_ctl, delta_s)
        
        if not mute:
            sys.stdout.write('\rChannel migration progress:%.2f%%' % ((it + 1) / n_iter * 100))
    if not mute:
        sys.stdout.write('\n')
    
    return X_ctl, Y_ctl, Z_ctl


def initialize_centerline(s_init, W, delta_s, n_bends, perturb, 
                          ypos=None, zpos=None):
    """
    Initialize river centerline. Assuming x is the longitudinal flow direction. First create a straight river
    centerline, then add perturbation to make it bended.
    :param s_init: (Float) - Length of the straight centerline.
    :param W: (Float) - Channel width.
    :param delta_s: (Float) - Distance between two consecutive points along centerline.
    :param n_bends: (Integer) - Number of bends in the centerline.
    :param perturb: (List) - Range of perturbation amplitude.
    :param ypos: (float): y-coordinate of the centerline's first point.
    :param zpos: (float): z-coordinate of the centerline.
    :return: x: (numpy.1darray) - x-coordinates of the initial centerline.
             y: (numpy.1darray) - y-coordinates of the initial centerline.
             z: (numpy.1darray) - z-coordinates of the initial centerline.
    """
    # x-coordinates of the centerline.
    x = np.arange(0, s_init + delta_s, delta_s, dtype=np.float32)
    # z-coordinates of the centerline.
    if zpos is None:
        z = np.zeros(len(x), dtype=np.float32)
    else:
        z = np.ones(len(x), dtype=np.float32) * zpos
    # Generate perturbation points.
    xp = np.linspace(0, s_init, n_bends + 2, dtype=np.float32)
    if ypos is None:
        yp = np.zeros(len(xp), dtype=np.float32)
    else:
        yp = np.ones(len(xp), dtype=np.float32) * ypos
    for i in range(1, len(yp) - 1):
        if perturb[1] <= 1:
            ptb = random.uniform(perturb[0], perturb[1]) * W
        if perturb[0] > 1:
            ptb = random.uniform(perturb[0], perturb[1])
        yp[i] += (-1) ** i * ptb
    # Interpolate bended centerline.
    interpolator = BiharmonicSpline2D(xp, yp)
    y = interpolator(x)
    return x, y, z


def initialize_centerline_V0(s_init, W, ypos, zpos, delta_s, n_bends, perturb):
    """
    Initialize river centerline. Assuming x is the longitudinal flow direction. First create a straight river
    centerline, then add perturbation to make it bended.
    :param s_init: (Float) - Length of the straight centerline.
    :param W: (Float) - Channel width.
    :param ypos: (Float) - y position of the centerline.
    :param zpos: (Float) - z position of the centerline.
    :param delta_s: (Float) - Distance between two consecutive points along centerline.
    :param n_bends: (Integer) - Number of bends in the centerline.
    :param perturb: (List) - Range of perturbation amplitude.
    :return: x: (numpy.1darray) - x-coordinates of the initial centerline.
             y: (numpy.1darray) - y-coordinates of the initial centerline.
             z: (numpy.1darray) - z-coordinates of the initial centerline.
    """
    # x-coordinates of the centerline.
    x = np.arange(0, s_init + delta_s, delta_s, dtype=np.float32)
    # Generate perturbation points.
    xp = np.linspace(0, s_init, n_bends + 2, dtype=np.float32)
    yp = np.ones(len(xp), dtype=np.float32) * ypos
    for i in range(1, len(yp) - 1):
        if perturb[1] <= 1:
            ptb = random.uniform(perturb[0], perturb[1]) * W
        if perturb[0] > 1:
            ptb = random.uniform(perturb[0], perturb[1])
        yp[i] += (-1) ** i * ptb
    # Interpolate bended centerline.
    interpolator = BiharmonicSpline2D(xp, yp)
    y = interpolator(x)
    z = np.ones(len(x), dtype=np.float32) * zpos
    return x, y, z


def resample_centerline(x, y, z, delta_s):
    """
    Re-sample centerline so that delta_s is roughly constant. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :param z: (numpy.1darray) - z-coordinates of centerline.
    :param delta_s: (Float) - Distance between two consecutive points along centerline.
    :return:
    """
    _, _, _, s = compute_curvelength(x, y)
    # Cubic spline interpolation. s=0 means no smoothing.
    tck = interpolate.splprep([x, y, z], s=0)
    unew = np.linspace(0, 1, 1 + int(round(s[-1] / delta_s)))
    out = interpolate.splev(unew, tck[0])
    x_res, y_res, z_res = out[0], out[1], out[2]
    x_res, y_res, z_res = x_res.astype(np.float32), y_res.astype(np.float32), z_res.astype(np.float32)
    return x_res, y_res, z_res


def compute_curvelength(x, y):
    """
    Compute the length of centerline. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :return: dx: (numpy.1darray) - First derivative of each point's x-coordinates on centerline.
             dy: (numpy.1darray) - First derivative of each point's y-coordinates on centerline.
             ds: (numpy.1darray) - The length of curve between two consecutive points along the centerline.
             s: (numpy.1darray) - Cumulated length of the curve.
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.hstack((0,np.cumsum(ds[1:])))
    return dx, dy, ds, s


def compute_curvature(x, y):
    """
    Compute the curvatures at each points of centerline. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :return: curvature: (numpy.1darray) - Curvatures at each points of centerline.
    """
    # First derivatives.
    dx = np.gradient(x)
    dy = np.gradient(y)
    # Second derivatives.
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
    return curvature


@numba.jit(nopython=True)  # Use numba to speed up the computation.
def compute_tangential_angle(x, y):
    """
    Compute tangential angle at each point of centerline.
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :return: beta: (numpy.1darray) - Tangential angle (radian) of each point.
    """
    beta = np.zeros(len(x), dtype=np.float32)  # Initialization.
    for i in range(len(x)):
        # The first point.
        if i == 0:
            if x[i + 1] == x[i]:  # Avoid division by zero.
                beta[i] = math.atan((y[i + 1] - y[i]) / 1e-6)
            else:
                beta[i] = math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            # The arc-tangent function can only return [-90? 90i, which means the angle in first quadrant is the same
            # as the angle in third quadrant, and the angle in second quadrant is the same as the angle in fourth
            # quadrant. The angles are in [-180? 180i through the process below.
            if y[i + 1] > y[i] and x[i + 1] < x[i]:
                beta[i] += math.pi
            if y[i + 1] < y[i] and x[i + 1] < x[i]:
                beta[i] -= math.pi
        # The end point.
        elif i == len(x) - 1:
            if x[i] == x[i - 1]:  # Avoid division by zero.
                beta[i] = math.atan((y[i] - y[i - 1]) / 1e-6)
            else:
                beta[i] = math.atan((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            # Angle transform.
            if y[i] > y[i - 1] and x[i] < x[i - 1]:
                beta[i] += math.pi
            if y[i] < y[i - 1] and x[i] < x[i - 1]:
                beta[i] -= math.pi
        # The interval points. Use three points (backward and forward) to compute the tangential angle.
        else:
            if x[i + 1] == x[i]:  # Avoid division by zero.
                beta_forward = math.atan((y[i + 1] - y[i]) / 1e-6)
            else:
                beta_forward = math.atan((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            if x[i] == x[i - 1]:  # Avoid division by zero.
                beta_backward = math.atan((y[i] - y[i - 1]) / 1e-6)
            else:
                beta_backward = math.atan((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            # Angle transform.
            if y[i + 1] > y[i] and x[i + 1] < x[i]:
                beta_forward += math.pi
            if y[i + 1] < y[i] and x[i + 1] < x[i]:
                beta_forward -= math.pi
            if y[i] > y[i - 1] and x[i] < x[i - 1]:
                beta_backward += math.pi
            if y[i] < y[i - 1] and x[i] < x[i - 1]:
                beta_backward -= math.pi
            beta[i] = 0.5 * (beta_forward + beta_backward)
            # This is the situation that the flow direction is opposite to the x-direction AND the middle point is
            # higher or lower than both forward point and backward point.
            if x[i + 1] < x[i - 1] and \
                    ((y[i] >= y[i + 1] and y[i] >= y[i - 1]) or (y[i] <= y[i - 1] and y[i] <= y[i + 1])):
                if beta[i] >= 0.0:
                    beta[i] -= math.pi
                else:
                    beta[i] += math.pi
    return beta


@numba.jit(nopython=True)  # Use numba to speed up the computation.
def compute_migration_rate(curv, ds, W, kl, Cf, D, pad_up, pad_down):
    """
    Compute migration rate of Howard-Knutson (1984) model. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param curv: (numpy.1darray) - Curvature of each point on centerline.
    :param ds: (numpy.1darray) - Distances between two consecutive points on centerline.
    :param W: (Float) - River's width.
    :param kl: (Float) - Migration constant (m/year).
    :param Cf: (Float) - Friction factor.
    :param D: (Float) - River's depth.
    :param pad_up: (Integer) - Number of points that will not migrate at upstream.
    :param pad_down: (Integer) - Number of points that will not migrate at downstream.
    :return: R1: (numpy.1darray) - The migration rate.
    """
    omega = -1.0
    gamma = 2.5
    k = 1.0
    R0 = kl * W * curv  # Nominal migration rate.
    R1 = np.zeros(len(R0), dtype=np.float32)  # Initialize adjusted migration rate.
    alpha = 2 * k * Cf / D
    if pad_up < 5:
        pad_up = 5
    for i in range(pad_up, len(R0) - pad_down):
        si = np.concatenate(
            (np.array([0]), np.cumsum(ds[i - 1::-1])))  # Cumulate distances backward from current point.
        G = np.exp(-alpha * si)
        # Adjusted migration rate in Howard-Knutson model.
        R1[i] = omega * R0[i] + gamma * np.sum(R0[i::-1] * G) / np.sum(G)
    return R1


def channel_bank(x, y, W):
    """
    Compute river banks' coordinates.
    :param x: (numpy.1darray) - x-coordinates of the centerline.
    :param y: (numpy.1darray) - y-coordinates of the centerline.
    :param W: (Float) - The channel's width.
    :return: xb: (numpy.2darray) - The x-coordinates of river banks. Shape: [len(x), 2].
                 Each row is the x-coordinates of two banks of a point in centerline.
             yb: (numpy.2darray) - The x-coordinates of river banks. Shape: [len(x), 2].
                 Each row is the y-coordinates of two banks of a point in centerline.
    """
    ns = len(x)
    angle = compute_tangential_angle(x, y)
    # Get the parabolas' endpoints' y-coordinates of each points on centerline.
    # Note that this is not the bank's y-coordinates until they are rotated.
    xb = np.c_[x, x]
    yb = np.c_[y - W / 2, y + W / 2]
    # Compute the parabola.
    for i in range(ns):
        R = np.array([[math.cos(angle[i]), -math.sin(angle[i])],  # Rotation matrix
                      [math.sin(angle[i]), math.cos(angle[i])]])
        [xb[i, :], yb[i, :]] = R @ [xb[i, :] - x[i], yb[i, :] - y[i]]  # Rotate to normal direction.
        xb[i, :] += x[i]  # x-coordinates of the erosion surface.
        yb[i, :] += y[i]  # y-coordinates of the erosion surface.
    return xb, yb


def compute_centerline_distance(x, y, xmin, ymin, dx, nx, ny):
    """
    Rasterize centerline and compute distance to centerline on X-Y plane. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :param xmin: (Float) - Minimum x-coordinates of the area of interest.
    :param ymin: (Float) - Minimum y-coordinates of the area of interest.
    :param dx: (Float) - X & Y resolution.
    :param nx: (Integer) - Number of points on x-direction.
    :param ny: (Integer) - Number of points on y-direction.
    :return: dist: (numpy.2darray) - Distance to centerline on X-Y plane.
    """
    ctl_pixels = []
    for i in range(len(x)):
        px = int((x[i] - xmin) / dx)
        py = int((y[i] - ymin) / dx)
        if 0 <= px < nx and 0 <= py < ny:
            ctl_pixels.append((py, px))
    if len(ctl_pixels):
        # Rasterize centerline.
        img = Image.new(mode='RGB', size=(ny, nx), color='white')  # Background is white.
        draw = ImageDraw.Draw(img)
        draw.point(ctl_pixels, fill='rgb(0, 0, 0)')  # Center-line is black.
        # Transfer image to array.
        pix = np.array(img)
        ctl = pix[:, :, 0]
        ctl[ctl == 255] = 1  # Background is 1, centerline is 0.
        # Compute distance to centerline.
        dist_map = ndimage.distance_transform_edt(ctl)
        dist = dist_map * dx  # The real distance.
        dist.astype(np.float32)
    else:
        dist = np.ones((nx, ny), dtype=np.float32) * 1e10
    return dist


def erosion_surface(cl_dist, zpos, W, D, mode=1):
    """
    Create erosion surface.
    :param cl_dist: (numpy.2darray) - Distance from centerline on X-Y plane.
    :param zpos: (Float) - z-coordinate of the channel top.
    :param W: (Float) - River's width.
    :param D: (Float) - River's maximum depth.
    :param mode: (Int) - Erosion surface shape. 1: U-shape surface; 2: V-shape surface.
    :return: ze: (numpy.2darray) - z-coordinates of erosion surface.
    """
    # U-shape surface.
    if mode == 1:
        ze = zpos + 4 * D / W ** 2 * (W ** 2 / 4 - cl_dist ** 2)
        ze = ze.astype(np.float32)
    # V-shape surface.
    elif mode == 2:
        ze1 = zpos + 4 * D / W ** 2 * (W ** 2 / 4 - cl_dist ** 2)
        ze2 = zpos + D * np.exp(-(cl_dist ** 2) / (2 * (W / 5) ** 2))
        ze = np.minimum(ze1, ze2)
        ze = ze.astype(np.float32)
    
    return ze


def lag_surface(cl_dist, zpos, h_lag, D):
    """
    Create Riverbed lag deposit surface.
    :param cl_dist: (numpy.2darray) - Distance from centerline on X-Y plane.
    :param zpos: (FLoat) - z-coordinates of the channel top.
    :param h_lag: (Float) - The maximum thickness of lag deposit.
    :param D: (Float) - River's maximum depth.
    :return: zl: (numpy.2darray) - z-coordinates of lag deposit surface.
    """
    zl = (zpos + D - h_lag) * np.ones(shape=cl_dist.shape)
    zl = zl.astype(np.float32)
    return zl


def pointbar_surface(cl_dist, z, W, D):
    """
    Create Riverbed point-bar surface.
    :param cl_dist: (numpy.2darray) - Distance from centerline on X-Y plane.
    :param z: (numpy.1darray) - z-coordinates of centerline.
    :param W: (Float) - River's width.
    :param D: (Float) - River's depth.
    :return: zpb: (numpy.2darray) - z-coordinates of point-bar surface.
    """
    if len(z[z - z[0] != 0]):
        raise ValueError('Can not process centerline with changing z-coordinates.')
    zpb = z[0] + D * np.exp(-(cl_dist ** 2) / (2 * (W / 4) ** 2))
    zpb = zpb.astype(np.float32)
    return zpb


def levee_surface(cl_dist, h_levee, w_levee, W, tp):
    """
    Create natural levee surface.
    :param cl_dist: (numpy.2darray) - Distance from centerline on X-Y plane.
    :param h_levee: (Float) - The Maximum thickness of levee.
    :param w_levee: (Float) - The width of levee.
    :param W: (Float) - River's width.
    :param tp: (numpy.2darray) - Topography.
    :return: zlv: (numpy.2darray) - z-coordinates of levee surface.
    """
    # th_levee = np.zeros(shape=cl_dist.shape, dtype=np.float32)
    # th_levee[cl_dist >= W/2] = -2 * h_levee / w_levee * (cl_dist[cl_dist >= W/2] - W / 2 - w_levee / 2)
    th1 = -2 * h_levee / w_levee * (cl_dist - W / 2 - w_levee / 2)
    th2 = np.ones(shape=cl_dist.shape) * h_levee
    th1, th2 = th1.astype(np.float32), th2.astype(np.float32)
    th_levee = np.minimum(th1, th2)
    th_levee[th_levee < 0] = 0
    zlv = tp - th_levee
    return zlv


def kth_diag_indices(a, k):
    """
    Function for finding diagonal indices with k offset.
    [From https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices]
    :param a: (numpy.2darray) - The input array.
    :param k: (Integer) - The offset. For example, k=1 represents the diagonal elements 1 step below the main diagonal,
              k=-1 represents the diagonal elements 1 step above the main diagonal.
    :return: rows: (numpy.1darray) - The row indices.
             col: (numpy.1darray) - The column indices.
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols


def find_neck(x, y, delta_s, critical_dist, n_buffer=20):
    """
    Find the location of neck cutoff. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :param critical_dist: (Float) - The critical distance. Cutoff occurs when distance of two points on centerline is
                          shorter than (or equal to) the critical distance.
    :param delta_s: (Float) - Distance between two consecutive points on centerline.
    :param n_buffer: (Integer) - Number of buffer points, preventing that cutoff occurs where there's no bend.
    :return: ind1: (numpy.1darray) - Indexes of centerline coordinates array where the cutoffs start.
             ind2: (numpy.1darray) - Indexes of centerline coordinates array where the cutoffs end.
    """
    # Number of neighbors that will be ignored for neck search.
    n_ignore = int((critical_dist + n_buffer * delta_s) / delta_s)
    # Compute Euclidean distance between each pair of points on centerline.
    dist = distance.cdist(np.array([x, y], dtype=np.float32).T, np.array([x, y], dtype=np.float32).T,
                          metric='euclidean')
    # Set distances greater than critical distance to NAN.
    dist[dist > critical_dist] = np.NAN
    # Set ignored neighbors' distance to NAN.
    for i in range(-n_ignore, n_ignore + 1):
        rows, cols = kth_diag_indices(dist, i)
        dist[rows, cols] = np.NAN
    # Find where the distance is lower than critical distance.
    r, c = np.where(~np.isnan(dist))
    # Take only the points measured downstream.
    ind1 = r[np.where(r < c)[0]]
    ind2 = c[np.where(r < c)[0]]
    return ind1, ind2


def execute_cutoff(x, y, z, delta_s, critical_dist, offset=20):
    """
    Execute cutoff on centerline. Modified from Zoltan Sylvester's meanderpy.
    [https://github.com/zsylvester/meanderpy]
    :param x: (numpy.1darray) - x-coordinates of centerline.
    :param y: (numpy.1darray) - y-coordinates of centerline.
    :param z: (numpy.1darray) - z-coordinates of centerline.
    :param delta_s: (Float) - Distance between two consecutive points on centerline.
    :param critical_dist: (Float) - The critical distance. Cutoff occurs when distance of two points on the centerline is
                          shorter than (or equal to) the critical distance.
    :param offset: (Integer) - Number of points from the cutoff points to the centerline points that will be connected.
                               Default is 20.
    :return: xc: (List) - x-coordinates of cutoffs.
             yc: (List) - y-coordinates of cutoffs.
             zc: (List) - z-coordinates of cutoffs.
             x: (numpy.1darray) - x-coordinates of centerline after cutoff.
             y: (numpy.1darray) - y-coordinates of centerline after cutoff.
             z: (numpy.1darray) - z-coordinates of centerline after cutoff.
    """
    xc = []
    yc = []
    zc = []
    ind1, ind2 = find_neck(x, y, delta_s, critical_dist)
    while len(ind1) > 0:
        xc.append(x[ind1[0]:ind2[0]+1])  # x-coordinates of cutoffs.
        yc.append(y[ind1[0]:ind2[0]+1])  # y-coordinates of cutoffs.
        zc.append(z[ind1[0]:ind2[0]+1])  # z-coordinates of cutoffs.
        x = np.concatenate((x[:ind1[0]+1-offset], x[ind2[0]+offset:]))  # x-coordinates of centerline after cutoff.
        y = np.concatenate((y[:ind1[0]+1-offset], y[ind2[0]+offset:]))  # y-coordinates of centerline after cutoff.
        z = np.concatenate((z[:ind1[0]+1-offset], z[ind2[0]+offset:]))  # z-coordinates of centerline after cutoff.
        ind1, ind2 = find_neck(x, y, delta_s, critical_dist)
    return xc, yc, zc, x, y, z


def plot_channel2D(channel_obj, oxbow_obj, show_oxbow=False, show_pointbar=False, show_migration=True,
                   interval=10, frame_delay=100, save_iter=1, show_iter=None, title=None):
    """
    Plot channel migration on X-Y plane.
    :param channel_obj: (List) - The channel objects.
    :param oxbow_obj: (List) - The oxbow-lake objects.
    :param title: (String) - The figure title. Default is "Meandering River Migration".
    :param interval: (Integer) - Plot channel for every "interval" channels. Default is 10.
    :param show_oxbow: (Bool) - Whether to display the oxbow lakes. Default is False.
    :param show_pointbar: (Bool) - Whether to display pointbars. Default is False.
    :param show_migration: (Bool) - Whether to show the river migration progress as an animation. Default is True.
    :param frame_delay: (Integer) - Delay between animation frames in milliseconds. Default is 100ms.
    :param save_it: (Integer) - The channel centerline is saved for every "save_iter" iteration. Default is 1.
    """
    # Set figure parameters.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')
    ax.set_facecolor('springgreen')
    ax.set_xlabel('X(m)', fontsize=20)
    ax.set_ylabel('Y(m)', fontsize=20)
    ax.tick_params(labelsize=20)
    if show_migration:
        frames = []
    if show_oxbow:
        oxbow = []
    if show_iter is not None:
        n_channel = show_iter // save_iter
    else:
        n_channel = len(channel_obj)
    for i in range(0, n_channel, interval):
        xc, yc = channel_obj[i].x, channel_obj[i].y  # Get centerline coordinates.
        wc = channel_obj[i].W  # Get channel width.
        xb, yb = channel_bank(xc, yc, wc)  # Compute bank coordinates.
        # Make the banks a closed curve.
        xb = np.hstack((xb[:, 0], xb[:, 1][::-1]))
        yb = np.hstack((yb[:, 0], yb[:, 1][::-1]))
        if i + interval >= n_channel:  # The most recent channel.
            if show_migration:
                arg = [xb, yb, 'blue']
                if show_oxbow:
                    arg.extend(oxbow)
                frame = ax.fill(*arg, edgecolor='black', animated=True)
            else:
                ax.fill(xb, yb, 'blue', edgecolor='black', animated=False)
        else:  # Older channels.
            if show_migration:
                arg = [xb, yb, 'blue']
                if show_oxbow:
                    arg.extend(oxbow)
                frame = ax.fill(*arg, edgecolor='black', animated=True)
            else:
                if show_pointbar:
                    ax.fill(xb, yb, color='yellow', edgecolor='black', alpha=0.5, animated=False)
        # Check if cutoff happens before.
        if show_oxbow:
            if i > 0:
                for j in range((i-interval)*save_iter, i*save_iter, 1):
                    # Oxbow lake (centerline) coordinates, lists of [array1, array2, ...].
                    xo, yo = oxbow_obj[j].x, oxbow_obj[j].y  
                    wo = oxbow_obj[j].W  # Oxbow width.
                    if len(xo) > 0:  # Oxbow lake exists.
                        n_oxbow = len(xo)
                        for k in range(n_oxbow):
                            # Compute bank coordinates of oxbow-lakes.
                            xbo, ybo = channel_bank(xo[k], yo[k], wo)  
                            # Make the banks a closed curve.
                            xbo = np.hstack((xbo[:, 0], xbo[:, 1][::-1]))
                            ybo = np.hstack((ybo[:, 0], ybo[:, 1][::-1]))
                            oxbow.extend([xbo, ybo, 'royalblue'])
                            if show_migration:
                                arg = [xb, yb, 'blue']
                                arg.extend(oxbow)
                                frame = ax.fill(*arg, edgecolor='black', animated=True)
                            else:
                                ax.fill(xbo, ybo, facecolor='royalblue', edgecolor='black', animated=False)
        if show_migration:
            frames.append(frame)  
    if title is None:
        if show_migration:
            title = 'Meandering River Migration'
        else:
            title = 'Meandering River (%d Iteration)' % (i * save_iter)
    ax.set_title(title, fontsize=26)
    if show_migration:
        anim = animation.ArtistAnimation(fig, frames, 
                                         interval=frame_delay, 
                                         blit=True, 
                                         repeat=True, 
                                         repeat_delay=1000)
    # plt.show()


def plot_distmap(dist, extent=None):
    """
    Plot distance map.

    Args:
        dist (Numpy.ndarray): distance map.
        extent (List): Map extent, in the format of [xmin, xmax, ymin, ymax].
                       Defaults to None, which is to use the number of samples.

    Returns:
        fig (Object): Matplotlib Figure object.
        ax (Object): Matplotlib Axes object.
    """
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.tick_params(labelsize=12)
    im = ax.imshow(dist.T, cmap='jet', aspect='equal', origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.set_label('Distance (m)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    
    return fig, ax


def plot_centerline(x, y, fig=None, ax=None):
    """
    Plot river channel centerline.

    Args:
        x (Numpy.ndarray): x-coordinates of the centerline.
        y (Numpy.ndarray): y-coordiantes of the centerline.
        fig (Object): Matplotlib Figure object.
        ax (Object): Matplotlib Axes object.
        
    Returns:
        fig (Object): Matplotlib Figure object.
        ax (Object): Matplotlib Axes object.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(21, 9))   
    ax.axis('equal')
    ax.set_xlabel('X(m)', fontsize=20)
    ax.set_ylabel('Y(m)', fontsize=20)
    ax.set_title('Channel Centerline', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.plot(x, y, 'ko', lw=2)
    fig.tight_layout()
    
    return fig, ax


def extract_numbers(string: str):
    """
    Extract the first number in a string.

    Args:
        string (str): String with number.

    Returns:
        number (int): The first number in the string.
    """
    # Find all the numbers in the string using regex
    match = re.findall(r'\d+', string)
    
    # Convert the matched strings to integers and add them to the numbers list
    number = int(match[0])
    
    return number


def get_database(database_dir: str, 
                 database_name: str, 
                 W: int):
    """
    Select database according to W value (river width).

    Args:
        database_dir (str): Database directory.
        database_name (str): Database name.
        W (int): W value in the name of database.

    Returns:
        my_database(str): The selected database.
    """
    # If the channel database name is a list and not empty.
    if isinstance(database_name, list) and len(database_name):
        
        # If the first element is None, then randomly choose a database, 
        # which satisfy "W < W_database", from the database directory.
        if database_name[0] is None:
            matches = []  # Databases that satisfy the condition.
            database_all = os.listdir(database_dir)
            for x in database_all:
                if x.startswith('.'):  # Remove hidden file/folders.
                    database_all.remove(x)
            for j in range(len(database_all)):
                W_database = extract_numbers(database_all[j])
                if W < W_database:
                    matches.append(database_all[j])
            if len(matches) == 0:
                raise ValueError("None of the database from %s satisfy (W < W_database)," 
                                 "get W=%.2f but W_database=%s." % 
                                 (database_dir, W, database_all))
            else:
                my_database = random.choice(matches)  # Choose a database. 
                database_name.pop(0)  # Pop out the first element from the database name list.
        
        # If the first element is the name of a database.
        else:
            W_database = extract_numbers(database_name[0])
            if W < W_database:
                my_database = database_name[0]
                database_name.pop(0)
            else:
                raise ValueError("The database %s you choose does not satisfy (W < W_database)," 
                                 "get W=%.2f but W_database=%.2f" % 
                                 (database_name[0], W, W_database))
        
    # If the channel database name is a list but empty, or the channel database name is None.
    if (isinstance(database_name, list) and len(database_name) == 0) or (database_name is None):
        matches = []  # Databases that satisfy the condition.
        database_all = os.listdir(database_dir)
        for x in database_all:
            if x.startswith('.'):  # Remove hidden file/folders.
                database_all.remove(x)
        for j in range(len(database_all)):
            W_database = extract_numbers(database_all[j])
            if W < W_database:
                matches.append(database_all[j])
        if len(matches) == 0:
            raise ValueError("None of the database from %s satisfy (W < W_database), " 
                             "get W=%.2f but W_database=%s." % 
                             (database_dir, W, database_all))
        else:
            my_database = random.choice(matches)  # Choose a database.
    
    # If the channel database name is a string.
    if isinstance(database_name, str):
        W_database = extract_numbers(database_name)
        if W < W_database:
            my_database = database_name
        else:
            raise ValueError("The database %s you choose does not satisfy (W < W_database), "
                             "get W=%.2f but W_database=%.2f" % 
                             (database_name, W, W_database)) 

    return my_database


def channel_vertical_trajectory(dt: float = 0.1, 
                                rt: float = 0.5, 
                                n_inci: int = 3, 
                                n_aggr: int = 3):
    # Number of time segments.
    nt = math.floor(1 / dt)  # All.
    nti = math.ceil(nt * rt)  # Incision.
    nta = nt - nti  # Aggradation.
    # Segment indexes.
    sxi = np.linspace(0, nti, nti, endpoint=False, dtype=np.int32)  # Incision.
    sxa = np.linspace(nti, nt, nta, endpoint=False, dtype=np.int32)  # Aggradation.
    # Choose segments.
    si = np.random.choice(sxi, n_inci, replace=False)  # Incision.
    sa = np.random.choice(sxa, n_aggr, replace=False)  # Aggradation.
    # Sort ascendingly.
    si = np.sort(si)
    sa = np.sort(sa)
    # Assemble incision time segments.
    t_inci = []
    for x in si:
        a, b = x * dt, (x + 1) * dt
        t_inci.append([a, b])
    # Assemble aggradation time segments.
    t_aggr = []
    for x in sa:
        a, b = x * dt, (x + 1) * dt
        t_aggr.append([a, b])
    
    return t_inci, t_aggr


def padding3D(x: np.ndarray, 
              pad_up: int, 
              pad_down: int) -> np.ndarray:
    """
    Padding a 3d array upward and downward with zeros in the last axis.

    Args:
        x (np.ndarray): The input 3d array, with the shape of (nx, ny, nz)
        pad_up (int): Upward padding amount.
        pad_down (int): Downward padding amount.

    Returns:
        np.ndarray: Padded 3d array, with the shape of (nx, ny, nz+pad_up+pad_down)
    """
    
    nx, ny, nz = x.shape
    y = np.zeros((nx, ny, nz+pad_up+pad_down), dtype=x.dtype)
    new_nz = y.shape[-1]
    y[:, :, pad_up:new_nz-pad_down] = x.copy()
    
    return y


def depadding3D(x: np.ndarray, 
                pad_up: int, 
                pad_down: int) -> np.ndarray:
    """
    De-padding a 3d array, will remove its top "pad_up" slices and bottom "pad_down" slices.

    Args:
        x (np.ndarray): The input 3d array, with the shape of (nx, ny, nz)
        pad_up (int): Number of slices to be removed at the array top.
        pad_down (int): Number of slices to be removed at the array bottom.

    Returns:
        np.ndarray: De-padded array, with the shape of (nx, ny, nz-pad_up-pad_down).
    """
    
    nz = x.shape[-1]
    y = x[:, :, pad_up:nz-pad_down].copy()
    
    return y
