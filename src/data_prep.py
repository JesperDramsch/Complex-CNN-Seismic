import numpy as np
from skimage.util.shape import view_as_windows
from skimage.io import imsave
from scipy.signal import hilbert

# Load Data
# Get it here: https://github.com/olivesgatech/facies_classification_benchmark
# Original here: https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete
train_seismic = np.load("data/train/train_seismic.npy")

# Calculate Complex traces
train_hilbert = np.zeros_like(train_seismic, dtype=np.complex)
for x in range(train_hilbert.shape[0]):
    for y in range(train_hilbert.shape[1]):
        train_hilbert[x,y,:] = hilbert(train_seismic[x,y,:])

train_complex = train_hilbert
train_complex = train_hilbert-train_seismic

# Generate Patch Data
patch_size = 64 
patch_size = 64

stride = 8

patch_shape = (1, patch_size, patch_size)

real_data = view_as_windows(train_seismic, patch_shape, step=stride)
cmplx_data = view_as_windows(train_complex, patch_shape, step=stride)

# Train - Validation Split
p = .9
val_split = np.random.choice(a=[False, True], size=real_data.shape[0:3], p=[p, 1-p])

# Inline Data
real = []
cmplx = []

for a in range(real_data.shape[0]):
    for b in range(real_data.shape[1]):
        for c in range(real_data.shape[2]):
            real.append(np.squeeze(real_data[a,b,c,0,:,:]).T)
            cmplx_patch = np.squeeze(cmplx_data[a,b,c,0,:,:]).T
            cmplx.append(np.stack([np.real(cmplx_patch), np.imag(cmplx_patch)], axis=2))

np.save('patch_data/i_real.npy', real)
np.save('patch_data/i_cmplx.npy', cmplx)

# Crossline Data
patch_shape = (patch_size, 1, patch_size)

real_data = view_as_windows(train_seismic, patch_shape, step=stride)
cmplx_data = view_as_windows(train_complex, patch_shape, step=stride)

real = []
cmplx = []

for a in range(real_data.shape[0]):
    for b in range(real_data.shape[1]):
        for c in range(real_data.shape[2]):
            real.append(np.squeeze(real_data[a,b,c,:,0,:]).T)
            cmplx_patch = np.squeeze(cmplx_data[a,b,c,:,0,:]).T
            cmplx.append(np.stack([np.real(cmplx_patch), np.imag(cmplx_patch)], axis=2))

np.save('patch_data/x_real.npy', real)
np.save('patch_data/x_cmplx.npy', cmplx)
