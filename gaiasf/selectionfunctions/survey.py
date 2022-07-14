from pathlib import Path
import h5py
import numpy as np
import xarray as xr


from abc import ABC, abstractmethod
import healpy as hp
from pathlib import Path
import pickle
from scipy.special import logit, expit
import xarray as xr
import numpy as np
import h5py

from gaiasf import utils, fetch_utils

# __all__ = ["validate_ds", "SelectionFunctionBase", ]


# Selection functions ported from gaiaverse's work ----------------
class DR2SelectionFunction(fetch_utils.DownloadMixin):
    """DR2 selection function developed by the Gaiaverse team."""

    bibcode = "2020MNRAS.497.4246B"
    datafiles = {
        "cog_ii_dr2.h5": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/PDFOVC/NYV9DM"
    }

    def __init__(self):

        datafile = fetch_utils.get_datadir() / "cog_ii_dr2.h5"
        with h5py.File(datafile) as f:
            # NOTE: HEAL pixelisation is in ICRS in these files!
            # n_field heal order=12, nside=4096
            # neighbour_field heal order=10, nside=1024
            theta = f["t_theta_percentiles"][()]
            alpha = f["ab_alpha_percentiles"][()]
            beta = f["ab_beta_percentiles"][()]
            self.ds = xr.Dataset(
                data_vars=dict(
                    theta=(["logrho", "g", "perc"], theta),
                    alpha=(["logrho", "g", "perc"], alpha),
                    beta=(["logrho", "g", "perc"], beta),
                    n_field=("ipix12", f["n_field"][()].astype(np.uint16)),
                    neighbour_field=(["ipix10"], f["neighbour_field"][()]),
                ),
                coords=dict(
                    logrho=f["log10_rho_grid"][()],
                    g=f["g_grid"][()],
                ),
            )

    def query(self):
        pass


class DR3SelectionFunction(DR2SelectionFunction):
    """DR3 selection function developed by the Gaiaverse team.

    This selection function assumes the same detection probabilty and simply
    swaps the number of scans map. The map resolution also changed from HEAL
    order 12 (nside=4096) to order 10 (nside=1024).
    """

    bibcode = ""  # TODO
    datafiles = DR2SelectionFunction.datafiles.copy()
    datafiles.update(
        {"n_field_dr3.h5": "https://dataverse.harvard.edu/api/access/datafile/4204267"}
    )

    def __init__(self, *args, **kwargs):
        super(DR3SelectionFunction, self).__init__()

        datafile = fetch_utils.get_datadir() / "n_field_dr3.h5"
        with h5py.File(datafile) as f:
            self.ds["n_field"] = ("ipix10", f["n_field"][()].astype(np.uint16))
