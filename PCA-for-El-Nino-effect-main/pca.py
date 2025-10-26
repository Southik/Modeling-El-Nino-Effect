import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat

data = loadmat('SSTPac.mat', squeeze_me=True)
PSSTA = data['PSSTA']
lat = data['lat']
lon = data['lon']

nlat = len(np.unique(lat)) 
nlon = len(np.unique(lon)) 

#check number of grid points in each direction is consistent
assert nlat * nlon == len(PSSTA[:, 0]), "Mismatch between spatial points and grid dimensions!"

cmap = plt.cm.RdBu_r  
cmap.set_bad(color='k')  

S = PSSTA.copy()

S[np.isnan(S)] = 0

S = S - np.mean(S, axis=1, keepdims=True)#mean free

U, s, Vt = np.linalg.svd(S, full_matrices=False)
variance = s**2
variance_fraction = variance / np.sum(variance)
fig, ax = plt.subplots()

####### TODO: Plot for Problem 1 #######
ax.plot(range(1, 21), variance_fraction[:20], 'o-')
ax.set_xlabel('Mode number')
ax.set_ylabel('Fraction of variance')
ax.set_title('Fraction of Variance Explained by Each Mode (Ex. 1)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

####### TODO: First plot for Problem 2 #######
EOF1 = U[:, 0]  # first EOF
PC1 = Vt[0, :]  # first principal component

fig, ax = plt.subplots()
sc = ax.scatter(lon, lat, c=EOF1, cmap=cmap, s=10)
plt.colorbar(sc, ax=ax, label='EOF1 Amplitude')
ax.set_title('Most Important Pattern (EOF1) (Ex.2)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig, ax = plt.subplots()

####### TODO: Second plot for Problem 2 #######
ax.plot(PC1)
ax.set_title('Principal Component of EOF1 (Ex. 2)')
ax.set_xlabel('Time index')
ax.set_ylabel('PC1 value')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

####### TODO: Code for Problem 3
std_PC1 = np.std(PC1)
elnino_indices = np.where(PC1 > std_PC1)[0]
lanina_indices = np.where(PC1 < -std_PC1)[0]

ElNino_mean = np.mean(S[:, elnino_indices], axis=1)
LaNina_mean = np.mean(S[:, lanina_indices], axis=1)

fig, (ax1, ax2) = plt.subplots(2,1)
sc1 = ax1.scatter(lon, lat, c=ElNino_mean, cmap=cmap, s=10)
plt.colorbar(sc1, ax=ax1, label='SSTA')
ax1.set_title('La Nina (PC1 < -1)')

sc2 = ax2.scatter(lon, lat, c=LaNina_mean, cmap=cmap, s=10)
plt.colorbar(sc2, ax=ax2, label='SSTA')
ax2.set_title('El Nino (PC1 > 1)')

ax2.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax2.set_ylabel('Latitude')

"""
Comment: 

In El Niño, the central and eastern Pacific gets warmer, 
while in La Niña, the same regions get cooler. 
The two patterns are roughly opposite, as expected, 
but they don't match perfectly—La Niña seems stronger in some areas.
This means the data is almost linear, but there are some differences, 
possibly due to more complex weather patterns.
"""

####### TODO: Plot for Problem 4 #######
plt.figure()
plt.hist(PC1, bins=30, density=True)
plt.title('Histogram of the First Principal Component (Ex. 4)')
plt.xlabel('PC1 value')
plt.ylabel('Probability density')

plt.show()
