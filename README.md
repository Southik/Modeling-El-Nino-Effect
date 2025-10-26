# PCA and Data Visualization Project

## Overview
This project applies **Principal Component Analysis (PCA)** to explore and visualize patterns in a dataset, verify PCA assumptions, and create a movie showing data reconstruction using leading modes.

---

## Files

### `pca.py`
Implements the PCA workflow:

1. **Data Preparation** – Read the dataset, subtract the mean to make it mean-free.  
2. **SVD Computation** – Perform singular value decomposition to obtain principal components.  
3. **Explained Variance Plot** – Plot the fraction of total variance explained by the top 20 modes.  
4. **Dominant Mode Visualization** – Plot the most important spatial pattern and its principal component time series.  
5. **El Niño / La Niña Analysis** – Identify frames with the first PC > +1σ (El Niño) and < –1σ (La Niña), plot their averages, and compare.  
6. **Unimodality Check** – Plot a histogram of the first principal component to confirm PCA’s suitability.

---

### `movie.py`
Generates an **MP4 video** showing:

- **Top panel:** Original data  
- **Bottom panel:** Data reconstructed using the first three modes  

Completed missing code and ensured clear labeling and smooth animation.

---

## Results
- The first few modes capture most variance.  
- El Niño and La Niña patterns are roughly opposite, supporting linearity.  
- The first PC is unimodal, validating PCA use.  
- The reconstructed data captures main structures of the original.

---

## Requirements
Install dependencies:
```bash
pip install numpy matplotlib scipy moviepy
