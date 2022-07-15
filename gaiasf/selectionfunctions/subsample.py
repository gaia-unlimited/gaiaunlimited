import pickle

import h5py
import healpy as hp
import numpy as np
import xarray as xr
from scipy.special import expit, logit

from gaiasf import fetch_utils, utils


def validate_ds(ds):
    """Validate if xarray.Dataset contains the expected selection function data."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("ds must be an xarray.Dataset.")
    for required_variable in ["logitp"]:
        if required_variable not in ds:
            raise ValueError(f"missing required variable {required_variable}.")
    diff = set(ds["logitp"].dims) - set(["g", "c", "ipix"])
    if diff:
        raise ValueError(f"Unrecognized dims of probability array: {diff}")


# TODO: spec out base class and op for combining SFs
class SelectionFunctionBase:
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

    def __init__(self, ds):
        validate_ds(ds)
        self.ds = ds

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

    @classmethod
    def from_conditions(cls, conditions):
        # potential idea
        # TODO: query for conditions and make Dataset from result
        pass

    def plot(self, *args, **kwargs):
        pass

    def query(self, coords, **kwargs):
        ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        factors = set(self.ds["p"].dims) - set({"ipix"})
        d = {}
        for k in factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = kwargs[k]
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        print(ipix, d)
        out = self.ds["p"].interp(ipix=ipix, **d)
        return out.to_numpy()


class DR3RVSSelectionFunction(SelectionFunctionBase):
    """Internal selection function for the RVS sample in DR3.

    This function gives the probability
        P(has RV | has G and G_RP)
    as a function of G magnitude and G-RP color.
    """

    def __init__(self):
        datadir = fetch_utils.get_datadir()
        with open(datadir / "wsdb-dr3-rvs-nk-g0_2-c0_4.pickle", "rb") as f:
            dr3 = pickle.load(f)
        df = dr3["df"].copy()
        df = df.loc[df["i_g"] <= 85]
        df["p"] = (df["k"] + 1) / (df["n"] + 2)
        df["logitp"] = logit(df["p"])
        dset_dr3 = xr.Dataset.from_dataframe(df.set_index(["ipix", "i_g", "i_c"]))
        gcenters, ccenters = dr3["g_mid"], dr3["c_mid"]
        dset_dr3 = dset_dr3.assign_coords(i_g=gcenters, i_c=ccenters)
        dset_dr3 = dset_dr3.rename({"i_g": "g", "i_c": "c"})
        super().__init__(dset_dr3)


# Selection functions ported from gaiaverse's work ----------------
class EDR3RVSSelectionFunction(SelectionFunctionBase, fetch_utils.DownloadMixin):
    """Internal selection function for the RVS sample in EDR3.

    This has been ported from the selectionfunctions by Gaiaverse team.

    NOTE: The definition of the RVS sample is not the same as DR3RVSSelectionFunction.
    """

    datafiles = {
        "rvs_cogv.h5": "https://dataverse.harvard.edu/api/access/datafile/5203267"
    }

    # frame = ''

    def __init__(self):
        with h5py.File(self._get_data("rvs_cogv.h5")) as f:
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
            ds = xr.Dataset(
                data_vars=dict(logitp=(["g", "c", "ipix"], x)),
                coords=dict(g=gcenters, c=ccenters),
            )
        super().__init__(ds)
