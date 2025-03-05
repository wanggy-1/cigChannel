"""
Generate topography from water discharge data.

Created by Guangyu Wang @ USTC
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage


def erosion_surface(distance, width, depth):
	
	z = 4 * depth / width**2 * (width**2 / 4 - distance**2)

	return z


# Params.
dx = 25  # Grid size.
thr = 0.2  # Binarization threshold.
mw_range = [200, 500]  # River maximum width range.
wdr_range = [10, 12]  # River width/depth ratio range.
idir = './discharge256'  # Input directory.
odir = './topography256'  # Output directory.
seed = None  # Random seed.

random.seed(seed)

for i in range(2820, 3001):
	print('Generating topo %d' % i)
	# Load water discharge data.
	img = Image.open(idir + f'/{i}.tiff')
	dc = np.array(img)  # Water discharge.

	# Water mask.
	msk = dc.copy()
	msk[msk < thr] = 0  # Non-water.
	msk[msk >= thr] = 1  # Water.
	msk = ndimage.gaussian_filter(msk, sigma=2)
	msk[msk < thr] = 0  # Non-water.
	msk[msk >= thr] = 1  # Water.

	# Distance map.
	dst, inds = ndimage.distance_transform_edt(1 - msk, return_indices=True)
	dst *= dx
	dstg = ndimage.gaussian_filter(dst, sigma=2)

	# Relative river width.
	m, n = inds[0, :, :], inds[1, :, :]
	w = dc[m, n]
	wg = ndimage.gaussian_filter(w, sigma=2)

	# Topography.
	mw = random.uniform(mw_range[0], mw_range[1])  # Maximum river width.
	wdr = random.uniform(wdr_range[0], wdr_range[1])  # River width/depth ratio.
	md = mw / wdr  # Maximum river depth.
	z = erosion_surface(distance=dst, 
						width=wg*mw, 
						depth=md)
	z = np.clip(z, 0, None)
	zg = ndimage.gaussian_filter(z, sigma=2)

	# Save topography.
	ofd = odir + '/' + str(i)  # Folder.
	if not os.path.exists(ofd):
		os.makedirs(ofd)
	np.save(ofd + '/data.npy', zg)

	# Plot.
	fig, ax = plt.subplots(2, 4, figsize=(15, 7.3))

	# Water discharge.
	im0 = ax[0, 0].imshow(dc, cmap='jet')
	ax[0, 0].set_title('Water Discharge')
	ax[0, 0].set_xticks([])
	ax[0, 0].set_yticks([])
	divider0 = make_axes_locatable(ax[0, 0])
	cax0 = divider0.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb0 = fig.colorbar(im0, cax=cax0, orientation='vertical')

	# Binary image (water mask).
	im1 = ax[0, 1].imshow(msk, cmap='gray')
	ax[0, 1].set_title('Binary Water Mask')
	ax[0, 1].set_xticks([])
	ax[0, 1].set_yticks([])
	divider1 = make_axes_locatable(ax[0, 1])
	cax1 = divider1.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')

	# Distance map (to water).
	im2 = ax[0, 2].imshow(dst, cmap='jet')
	ax[0, 2].set_title('Distance Map')
	ax[0, 2].set_xticks([])
	ax[0, 2].set_yticks([])
	divider2 = make_axes_locatable(ax[0, 2])
	cax2 = divider2.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')

	# Smoothed distance map.
	im3 = ax[0, 3].imshow(dstg, cmap='jet')
	ax[0, 3].set_title('Smoothed Distance Map')
	ax[0, 3].set_xticks([])
	ax[0, 3].set_yticks([])
	divider3 = make_axes_locatable(ax[0, 3])
	cax3 = divider3.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb3 = fig.colorbar(im3, cax=cax3, orientation='vertical')

	# Relative river width.
	im4 = ax[1, 0].imshow(w, cmap='jet')
	ax[1, 0].set_title('Relative River Width')
	ax[1, 0].set_xticks([])
	ax[1, 0].set_yticks([])
	divider4 = make_axes_locatable(ax[1, 0])
	cax4 = divider4.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb4 = fig.colorbar(im4, cax=cax4, orientation='vertical')

	# Smoothed relative river width.
	im5 = ax[1, 1].imshow(wg, cmap='jet')
	ax[1, 1].set_title('Smoothed River Width')
	ax[1, 1].set_xticks([])
	ax[1, 1].set_yticks([])
	divider5 = make_axes_locatable(ax[1, 1])
	cax5 = divider5.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb5 = fig.colorbar(im5, cax=cax5, orientation='vertical')

	# Topography.
	im6 = ax[1, 2].imshow(z, cmap='terrain_r')
	ax[1, 2].set_title('Topography')
	ax[1, 2].set_xticks([])
	ax[1, 2].set_yticks([])
	divider6 = make_axes_locatable(ax[1, 2])
	cax6 = divider6.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb6 = fig.colorbar(im6, cax=cax6, orientation='vertical')

	# Smoothed topography.
	im7 = ax[1, 3].imshow(zg, cmap='terrain_r')
	ax[1, 3].set_title('Smoothed Topography')
	ax[1, 3].set_xticks([])
	ax[1, 3].set_yticks([])
	divider7 = make_axes_locatable(ax[1, 3])
	cax7 = divider7.append_axes(position='right', 
								size=0.1, 
								pad=0.1)
	cb7 = fig.colorbar(im7, cax=cax7, orientation='vertical')

	fig.tight_layout()

	# Save figure.
	fig.savefig(ofd + '/pic.png', dpi=96)
	
	# Close figure.
	plt.close()
