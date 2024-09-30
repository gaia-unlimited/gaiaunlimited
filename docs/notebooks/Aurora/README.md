# Selection function for metal-poor giants in the heart of the Galaxy

<!-- URL: [https://github.com/evgenykurbatov/kp23-turb-conv-ppd](https://github.com/evgenykurbatov/kp23-turb-conv-ppd) -->

This is a part of the [_GaiaUnlimited_](https://github.com/gaia-unlimited) project.

<!-- [![arxiv](http://img.shields.io/badge/astro.GA-arXiv%3A2106.07653-B31B1B.svg)](https://arxiv.org/abs/2106.07653) -->


## Summary

We utilize red giant branch (RGB) stars from *Gaia*, with metallicities estimated by Andrae et al. (2023) using spectro-photometry from *Gaia* Data Release 3 (XP). By accounting for *Gaia*'s selection functions and testing several parametric density models, we examine the spatial distribution of metal-poor ([M/H]<-1.3) RGB stars, from the Galactic center (r ~ 1 kpc) out to beyond the Solar radius (r ~ 20 kpc).

This is a worked example of a specialized selection function for the Gaia survey, based on the forthcoming paper by Evgeney Kurbatov et al.


## Installation

Install the requirements with:

    pip install -r requirements.txt


## Code

The whole pipeline is:

- `1. Download and prepare RGB star catalogue.ipynb`
  - Download the Andrae R. et al. (2023) catalogue of RGB stars
  - Make the transformation from Galactic to Galactocentric frame
- `1a. Plot kinematics.ipynb` (optional)
  - Clean the RGB star catalogue of globular clusters (Vasiliev & Baumgardt 2021), SMC and LMC
  - Use AGAMA for potential estimate
  - Make cool plots
- `2. Extinctions.ipynb`
  - Estimate extinctions in G, RP and, BP bands
  - Estimate monochromatic exctinction A_0
  - Fit A_G(A_0) A_BP(A_0) and A_RP(A_0) neglecting the T_eff dependency
- `3. Luminosity function.ipynb`
  - Take NGC 6397 globular cluster from Vasiliev & Baumgardt (2021) catalogue
  - Correct the magnitudes for extinctions
  - Extract luminosity function of the RGB stars
- `4. Parallax errors.ipynb`
  - Query and count Gaia source (GS) stars on the HEALPix vs G grid
  - Count XP stars on the HEALPix vs G grid
  - Count number of visits and bin parallax errors
  - Fit the parallax error vs G magnitude
- `5. Domain.ipynb`
  - Define the computational domain
  - Calculate map from (l, b, D) to (x, y, z)
- `6. XP selection function.ipynb`
  - Query XP statistics over the sky and the G band
  - Estimate subsample selection function
- `7. Probability transformation function.ipynb`
  - Join together the extinction A_G, the XP selection function, the parallax error model and the luminosity function
  - Estimate the transformation function for PDF, from model to observable
- `8. Aurora model.ipynb`
  One example of the Aurora + GS/E model
