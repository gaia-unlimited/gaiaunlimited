import h5py
import healpy as hp
import numpy as np
import xarray as xr
from scipy import stats

from gaiasf import fetch_utils, utils

__all__ = ["DR2SelectionFunction", "DR3SelectionFunction"]


# Selection functions ported from gaiaverse's work ----------------
class DR2SelectionFunction(fetch_utils.DownloadMixin):
    """DR2 selection function developed by the Gaiaverse team."""

    bibcode = "2020MNRAS.497.4246B"
    datafiles = {
        "cog_ii_dr2.h5": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/PDFOVC/NYV9DM"
    }
    nside_n_field = 4096
    nside_crowding = 1024

    def __init__(self):

        with h5py.File(self._get_data("cog_ii_dr2.h5")) as f:
            # NOTE: HEAL pixelisation is in ICRS in these files!
            # n_field heal order=12, nside=4096
            # neighbour_field heal order=10, nside=1024
            theta = f["t_theta_percentiles"][()]
            alpha = f["ab_alpha_percentiles"][()]
            beta = f["ab_beta_percentiles"][()]
            ds = xr.Dataset(
                data_vars=dict(
                    theta=(["logrho", "g", "perc"], theta),
                    alpha=(["logrho", "g", "perc"], alpha),
                    beta=(["logrho", "g", "perc"], beta),
                    n_field=("ipix", f["n_field"][()].astype(np.uint16)),
                    neighbour_field=(["ipix_logrho"], f["neighbour_field"][()]),
                ),
                coords=dict(
                    logrho=f["log10_rho_grid"][()],
                    g=f["g_grid"][()],
                ),
            )
        # Add logrho
        ds["logrho_field"] = np.log10(
            np.maximum(1.0, ds["neighbour_field"])
            / hp.nside2pixarea(self.nside_crowding, degrees=True)
        )
        self.ds = ds

    def query(self, coords, gmag, use_modelT=False):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            gmag (float or array): G magnitudes. Should have the same shape as coords.

        Returns:
            prob: array of selection probabilities.
        """
        if coords.shape != np.shape(gmag):
            raise ValueError(f"Input shape mismatch: {coords.shape} != {gmag.shape}")
        ipix = utils.coord2healpix(coords, "icrs", self.nside_n_field)
        ipix_logrho = utils.coord2healpix(coords, "icrs", self.nside_crowding)
        logrho = self.ds["logrho_field"].sel(ipix_logrho=ipix_logrho).to_numpy()

        # NOTE: xr.DataArray's interp method calls scipy's interp1d or interpnd
        # depending on the input shape. The two methods do not have a consistent
        # keyword for `fill_value` when we want to force extrapolation.
        # For interp1d, it is 'extrapolate', for interpnd it is None.
        # Let's make input at least 1d array so that we can not worry about this.
        ipix = np.atleast_1d(ipix)
        gmag = np.atleast_1d(gmag)
        logrho = np.atleast_1d(logrho)
        ns = self.ds["n_field"].sel(ipix=ipix).to_numpy()
        kwargs = dict(
            method="nearest",
            kwargs=dict(fill_value=None),
        )
        if use_modelT:
            ps = (
                self.ds["theta"]
                .sel(perc=2)
                .interp(g=xr.DataArray(gmag), logrho=xr.DataArray(logrho), **kwargs)
                .to_numpy()
            )
            out = stats.binom(ns, ps).sf(4)
        else:
            alphas = (
                self.ds["alpha"]
                .sel(perc=2)
                .interp(g=xr.DataArray(gmag), logrho=xr.DataArray(logrho), **kwargs)
                .to_numpy()
            )
            betas = (
                self.ds["beta"]
                .sel(perc=2)
                .interp(g=xr.DataArray(gmag), logrho=xr.DataArray(logrho), **kwargs)
                .to_numpy()
            )
            out = stats.betabinom(ns, alphas, betas).sf(4)

        if len(coords.shape) == 0:
            return out.squeeze()
        else:
            return out


# TODO: I guess we will replace this?
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
    nside_n_field = 1024
    nside_crowding = 1024

    def __init__(self, *args, **kwargs):
        super(DR3SelectionFunction, self).__init__()

        with h5py.File(self._get_data("n_field_dr3.h5")) as f:
            self.ds["n_field"] = ("ipix", f["n_field"][()].astype(np.uint16))
