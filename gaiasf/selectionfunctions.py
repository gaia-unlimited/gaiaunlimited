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

__all__ = ["validate_ds", "SelectionFunctionBase"]


def validate_ds(ds):
    """Validate if xarray.Dataset contains the expected selection function data."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("ds must be an xarray.Dataset.")
    for required_coord in ["g", "ipix"]:
        if required_coord not in ds.coords:
            raise ValueError(f"missing required coordinate {required_coord}.")
    for required_variable in ["p", "logitp"]:
        if required_variable not in ds:
            raise ValueError(f"missing required variable {required_variable}.")


# TODO: spec out base class and op for combining SFs
class SelectionFunctionBase(ABC):
    """Base class for Gaia selection functions.

    Selection function is defined as the selection probability as a function of
    Gaia G magnitude and healpix location, and optionally also on G-G_PR color.

    We use xarray.Dataset as the main data structure to contain this
    multi-dimensional map of probabilities. This Dataset instance should be
    attach as `.ds` and have the following schema:
        - must contain data variable `p` and `logitp` for selection probability
        and logit of that selection probability.
        - must have coodinates
            - ipix for healpix id in int64 dtype
            - g for Gaia G magnitude
            - c for Gaia G - G_RP color

    It is assumed that ipix is the full index array for the given healpix order
    with no missing index.
    """

    @property
    def order(self):
        return hp.npix2order(self.ds["ipix"].size)

    @property
    def nside(self):
        return hp.npix2nside(self.ds["ipix"].size)

    def __mul__(self, other):
        # if not isinstance(other, SelectionFunctionBase):
        #     raise TypeError()
        # TODO
        pass

    def coord2healpix(self, coords):
        # TODO: implement using coord2healpix
        pass

    def from_conditions(self, conditions):
        # potential idea
        # TODO: query for conditions and make Dataset from result
        pass

    def plot(self, *args, **kwargs):
        pass


class DR3RVSSelectionFunction(SelectionFunctionBase):
    """Internal selection function for the RVS sample in DR3.

    This function gives the probability
        P(has RV | has G and G_RP)
    as a function of G magnitude and G-RP color.
    """

    def __init__(self):
        tmpdir = Path("~/Work/GaiaUnlimited/notebooks").expanduser()
        with open(tmpdir / "wsdb-dr3-rvs-nk-g0_2-c0_4.pickle", "rb") as f:
            dr3 = pickle.load(f)
        df = dr3["df"].copy()
        df = df.loc[df["i_g"] <= 85]
        df["p"] = (df["k"] + 1) / (df["n"] + 2)
        df["logitp"] = logit(df["p"])
        dset_dr3 = xr.Dataset.from_dataframe(df.set_index(["ipix", "i_g", "i_c"]))
        gcenters, ccenters = dr3["g_mid"], dr3["c_mid"]
        dset_dr3 = dset_dr3.assign_coords(i_g=gcenters, i_c=ccenters)
        dset_dr3 = dset_dr3.rename({"i_g": "g", "i_c": "c"})
        self.ds = dset_dr3

    def query(
        self,
    ):
        pass


# Selection functions ported from gaiaverse's work ----------------
class EDR3RVSSelectionFunction(object):
    """Internal selection function for the RVS sample in EDR3.

    This has been ported from the selectionfunctions by Gaiaverse team.

    NOTE: The definition of the RVS sample is not the same as DR3RVSSelectionFunction.
    """

    # frame = ''

    def __init__(self):
        tmpdir = Path("/Users/soh/Work/GaiaUnlimited/gaiaverse/data/")
        datafile = tmpdir / "rvs_cogv.h5"
        with h5py.File(datafile) as f:
            # NOTE: HEAL pixelisation is in ICRS in these files!
            # x is the logit probability evaluated at each (mag, color, ipix)
            x = f["x"][()]  # dims: (n_mag_bins, n_color_bins, n_healpix_order6)
            # these are model parameters
            # b = f['b'][()]
            # z = f['z'][()]
            attrs = dict(f["x"].attrs.items())
            n_gbins, n_cbins, n_pix = x.shape
            gbins = np.linspace(*attrs["Mlim"], n_gbins + 1)
            cbins = np.linspace(*attrs["Clim"], n_cbins + 1)
            edges2centers = lambda x: (x[1:] + x[:-1]) * 0.5
            gcenters, ccenters = edges2centers(gbins), edges2centers(cbins)
            self.ds = xr.Dataset(
                data_vars=dict(logitp=(["g", "c", "ipix6"], x)),
                coords=dict(g=gcenters, c=ccenters),
            )

    def query(
        self,
    ):
        pass


class DR2SelectionFunction:
    """DR2 selection function developed by the Gaiaverse team."""

    bibcode = "2020MNRAS.497.4246B"

    def __init__(
        self,
    ):

        tmpdir = Path("~/Work/GaiaUnlimited/notebooks").expanduser()
        datafile = tmpdir / "../gaiaverse/data/cog_ii/cog_ii_dr2.h5"
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

    def __init__(self, *args, **kwargs):
        super(DR3SelectionFunction, self).__init__()

        tmpdir = Path("~/Work/GaiaUnlimited/notebooks").expanduser()
        datafile = tmpdir / "../gaiaverse/data/cog_ii/n_field_dr3.h5"
        with h5py.File(datafile) as f:
            self.ds["n_field"] = ("ipix10", f["n_field"][()].astype(np.uint16))
