import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import svd
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

data = loadmat('SSTPac.mat', squeeze_me=True)
PSSTA = data['PSSTA']
yd = data['yd']  
lat = np.unique(data['lat'])  
lon = np.unique(data['lon'])  

print("PSSTA shape:", PSSTA.shape)
print("Latitude array length:", len(lat))
print("Longitude array length:", len(lon))

nlat = len(lat)
nlon = len(lon)

#grid dimensions match the data
if nlat * nlon != PSSTA.shape[0]:
    raise ValueError(
        f"Mismatch in dimensions: nlat * nlon = {nlat * nlon}, but PSSTA.shape[0] = {PSSTA.shape[0]}"
    )

print("NaN values in PSSTA:", np.isnan(PSSTA).any())
print("Inf values in PSSTA:", np.isinf(PSSTA).any())

# Replace problematic values
PSSTA = np.nan_to_num(PSSTA, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize the data 
PSSTA = (PSSTA - np.mean(PSSTA)) / np.std(PSSTA)

U, S, VT = svd(PSSTA, full_matrices=False)
PSSTA_reconstructed = np.dot(U[:, :3], np.dot(np.diag(S[:3]), VT[:3, :]))

cmap = plt.cm.coolwarm
cmap.set_bad(color='k')  # NaN values as black

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
fig.subplots_adjust(hspace=0.4)

#plot original data
image1 = ax1.imshow(
    PSSTA[:, 0].reshape(nlat, nlon),
    cmap=cmap,
    aspect='auto',
    vmin=-3,
    vmax=3,
    interpolation='none',
    origin='lower',
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]
)
ax1.set_title('Original Data')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
fig.colorbar(image1, ax=ax1)

#plot reconstructed data
image2 = ax2.imshow(
    PSSTA_reconstructed[:, 0].reshape(nlat, nlon),
    cmap=cmap,
    aspect='auto',
    vmin=-3,
    vmax=3,
    interpolation='none',
    origin='lower',
    extent=[lon.min(), lon.max(), lat.min(), lat.max()]
)
ax2.set_title('Reconstructed Data (First 3 Modes)')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
fig.colorbar(image2, ax=ax2)

plt.close(fig)

duration = 30.0  
fps = int(len(yd) / duration)  

def make_frame_mpl(t):
    frame = int(t * fps)
    ax1.set_title(f'Original Data: Year = {int(yd[frame])}')
    image1.set_data(PSSTA[:, frame].reshape(nlat, nlon))
    
    ax2.set_title(f'Reconstructed Data: Year = {int(yd[frame])}')
    image2.set_data(PSSTA_reconstructed[:, frame].reshape(nlat, nlon))
    
    return mplfig_to_npimage(fig)

animation = mpy.VideoClip(make_frame_mpl, duration=duration)

animation.write_videofile('SSTA_animation.mp4', fps=fps)
