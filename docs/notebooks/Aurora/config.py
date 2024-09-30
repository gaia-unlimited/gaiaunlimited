
import numpy as np


cache_path = './cache/'

fig_path = './fig/'
fig_dpi = 300

# Use HEALPix 7 for extinction estimate and plotting
hpx_order = 7
hpx_base = 2**35 * 4**(12 - hpx_order)
hpx_nside = 2**hpx_order
hpx_npix = 12 * 4**hpx_order
print(f"hpx_order={hpx_order} --> (hpx_nside={hpx_nside}, hpx_npix={hpx_npix})")

# This pixelization level is used for the Model
model_hpx_order = 5
model_hpx_base = 2**35 * 4**(12 - model_hpx_order)
model_hpx_nside = 2**model_hpx_order
model_hpx_npix = 12 * 4**model_hpx_order
print(f"model_hpx_order={model_hpx_order} --> (model_hpx_nside={model_hpx_nside}, model_hpx_npix={model_hpx_npix})")

## Magnitude bins
G_min, G_max, dG  = 1.7, 23.5, 0.2  ## mag
G_bins = np.linspace(G_min,  G_max,  int((G_max-G_min)/dG) + 1)
