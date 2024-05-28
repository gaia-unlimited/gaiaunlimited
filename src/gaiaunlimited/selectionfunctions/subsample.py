from collections import OrderedDict
import h5py
import pandas as pd
import healpy as hp
import numpy as np
import xarray as xr
from scipy.special import expit, logit
from astroquery.gaia import Gaia
import ast

from .. import fetch_utils, utils

__all__ = [
    "SelectionFunctionBase",
    "DR3RVSSelectionFunction",
    "EDR3RVSSelectionFunction",
    "SubsampleSelectionFunction",
    "SubsampleSelectionFunctionHMLE",
]


def _validate_ds(ds):
    """Validate if xarray.Dataset contains the expected selection function data."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("ds must be an xarray.Dataset.")
    for required_variable in ["logitp"]:
        if required_variable not in ds:
            raise ValueError(f"missing required variable {required_variable}.")
    if ds.dims.keys() - set(["ipix"]) == {"g", "c"}:
        diff = set(ds["logitp"].dims) - set(["g", "c", "ipix"])
    else:
        diff = set(ds["logitp"].dims) - ds.dims.keys()
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
        _validate_ds(ds)
        self.ds = ds

    @property
    def order(self):
        """Order of the HEALPix."""
        return hp.npix2order(self.ds["ipix"].size)

    @property
    def nside(self):
        """Nside of the HEALPix."""
        return hp.npix2nside(self.ds["ipix"].size)

    @property
    def factors(self):
        """Variables other than HEALPix id that define the selection function."""
        return set(self.ds["logitp"].dims) - set({"ipix"})

    def __mul__(self, other):
        # if not isinstance(other, SelectionFunctionBase):
        #     raise TypeError()
        # TODO
        raise NotImplementedError()

    @classmethod
    def from_conditions(cls, conditions):
        # potential idea
        # TODO: query for conditions and make Dataset from result
        raise NotImplementedError()

    def query(self, coords,return_variance = False,fill_nan = False, **kwargs):
        """Query the selection function at the given coordinates.

        Args:
            coords: sky coordinates as an astropy coordinates instance.

        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.

        Returns:
            np.array: array of internal selection probabilities.
        """
        # NOTE: make input atleast_1d for .interp keyword consistency.
        try:
            if list(self.datafiles)[0] == "dr3-rvs-nk.h5":
                ipix = utils.coord2healpix(coords, "galactic", self.nside, nest=True)
            else:
                ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        except:
            ipix = utils.coord2healpix(coords, "icrs", self.nside, nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))
        d = {}
        for k in self.factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d["method"] = "nearest"
        d["kwargs"] = dict(fill_value=None)  # extrapolates
        out = self.ds["logitp"].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()
        if return_variance:
            out_variance = self.ds["logit_p_variance"].interp(ipix=ipix, **d).to_numpy()
            out_variance.squeeze()
            if fill_nan:
                out = np.nan_to_num(out,nan = logit(1./2.))
                out_variance = np.nan_to_num(out_variance,nan = logit(1./12.))
            return expit(out),expit(out_variance)
        else:
            if fill_nan: out = np.nan_to_num(out,nan = logit(1./2.))
            return expit(out)


class DR3RVSSelectionFunction(SelectionFunctionBase, fetch_utils.DownloadMixin):
    """Internal selection function for the RVS sample in DR3.

    This function gives the probability

        P(has RV | has G and G_RP)

    as a function of G magnitude and G-RP color.
    """

    datafiles = {
        "dr3-rvs-nk.h5": "https://zenodo.org/record/8300616/files/dr3-rvs-nk.h5"
    }

    def __init__(self):
        with h5py.File(self._get_data("dr3-rvs-nk.h5")) as f:
            df = pd.DataFrame.from_records(f["data"][()])
            df["p"] = (df["k"] + 1) / (df["n"] + 2)
            df["logitp"] = logit(df["p"])
            dset_dr3 = xr.Dataset.from_dataframe(df.set_index(["ipix", "i_g", "i_c"]))
            gcenters, ccenters = f["g_mid"][()], f["c_mid"][()]
            dset_dr3 = dset_dr3.assign_coords(i_g=gcenters, i_c=ccenters)
            dset_dr3 = dset_dr3.rename({"i_g": "g", "i_c": "c"})
            super().__init__(dset_dr3)


class SubsampleSelectionFunction(SelectionFunctionBase):
    """Internal selection function for any sample of DR3.

    This function gives the probability

        P(is in subsample| is in Gaia and has G and G_RP)

    as a function of G magnitude and G-RP color.

    See example: https://gaiaunlimited.readthedocs.io/en/latest/notebooks/SubsampleSF_Tutorial.html

    If you use this module in a publication please cite::

        @ARTICLE{2023A&A...677A..37C,
               author = {{Castro-Ginard}, Alfred and {Brown}, Anthony G.~A. and {Kostrzewa-Rutkowska}, Zuzanna and {Cantat-Gaudin}, Tristan and {Drimmel}, Ronald and {Oh}, Semyeong and {Belokurov}, Vasily and {Casey}, Andrew R. and {Fouesneau}, Morgan and {Khanna}, Shourya and {Price-Whelan}, Adrian M. and {Rix}, Hans-Walter},
                title = "{Estimating the selection function of Gaia DR3 subsamples}",
              journal = {\aap},
             keywords = {Galaxy: general, methods: statistical, catalogs, Astrophysics - Astrophysics of Galaxies},
                 year = 2023,
                month = sep,
               volume = {677},
                  eid = {A37},
                pages = {A37},
                  doi = {10.1051/0004-6361/202346547},
        archivePrefix = {arXiv},
               eprint = {2303.17738},
         primaryClass = {astro-ph.GA},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...677A..37C},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
    """

    def __init__(self, subsample_query, file_name, hplevel_and_binning):
        def _download_binned_subset(self):
            if len(self.hplevel_and_binning) == 1:
                query_to_gaia = f"""SELECT {self.column_names_for_select_clause} COUNT(*) AS n, SUM(selection) AS k
                                    FROM (SELECT {self.binning}
                                        to_integer(IF_THEN_ELSE('{self.subsample_query}',1.0,0.0)) AS selection
                                        FROM gaiadr3.gaia_source) AS subquery
                                    GROUP BY {self.group_by_clause}"""
            else:
                query_to_gaia = f"""SELECT {self.column_names_for_select_clause} COUNT(*) AS n, SUM(selection) AS k
                                    FROM (SELECT {self.binning}
                                        to_integer(IF_THEN_ELSE('{self.subsample_query}',1.0,0.0)) AS selection
                                        FROM gaiadr3.gaia_source
                                        WHERE {self.where_clause.strip("AND ")}) AS subquery
                                    GROUP BY {self.group_by_clause}"""
            job = Gaia.launch_job_async(query_to_gaia, name=self.file_name)
            r = job.get_results()
            df = r.to_pandas()
            columns = [key + "_" for key in self.hplevel_and_binning.keys()]
            columns += ["n", "k"]
            with open(fetch_utils.get_datadir() / f"{self.file_name}.csv", "w") as f:
                f.write(f"#{self.hplevel_and_binning}\n")
                df[columns].to_csv(f, index = False)
            return df[columns]

        self.subsample_query = subsample_query
        self.file_name = file_name
        self.hplevel_and_binning = hplevel_and_binning
        self.column_names_for_select_clause = ""
        self.binning = ""
        self.where_clause = ""
        for key in self.hplevel_and_binning.keys():
            if key == "healpix":
                self.healpix_level = self.hplevel_and_binning[key]
                self.binning = (
                    self.binning
                    + f"""to_integer(GAIA_HEALPIX_INDEX({self.healpix_level},source_id)) AS {key+"_"}, """
                )
            else:
                self.low, self.high, self.bins = self.hplevel_and_binning[key]
                self.binning = (
                    self.binning
                    + f"""to_integer(floor(({key} - {self.low})/{self.bins})) AS {key+"_"}, """
                )
                self.where_clause = (
                    self.where_clause + f"""{key} > {self.low} AND {key} < {self.high} AND """
                )
            self.column_names_for_select_clause = self.column_names_for_select_clause + key + "_" + ", "
        self.group_by_clause = self.column_names_for_select_clause.strip(", ")

        if (fetch_utils.get_datadir() / f"{self.file_name}.csv").exists():
            with open(fetch_utils.get_datadir() / f"{self.file_name}.csv", "r") as f:
                params = f.readline()
            if self.hplevel_and_binning == ast.literal_eval(params.strip("#").strip("\n")):
                df = pd.read_csv(
                    fetch_utils.get_datadir() / f"{self.file_name}.csv",
                    comment="#",
                )
            else:
                df = _download_binned_subset(self)
        else:
            df = _download_binned_subset(self)

        columns = [key + "_" for key in self.hplevel_and_binning.keys()]
        columns += ["n", "k"]
        df = df[columns]
        df["p"] = (df["k"] + 1) / (df["n"] + 2)
        df["logitp"] = logit(df["p"])
        df["p_variance"] = (df["n"] + 1) * (df["n"] - df["k"] + 1) / (df["n"] + 2) / (df["n"] + 2) / (df["n"] + 3)
        df["logit_p_variance"] = logit(df["p_variance"])
        for key in self.hplevel_and_binning.keys():
            if key == 'healpix': continue
            lencol = len(np.unique(df[key+'_']))
            if len(np.arange(self.hplevel_and_binning[key][0],self.hplevel_and_binning[key][1],self.hplevel_and_binning[key][2])) == lencol: continue
            print('Empty slice in {}, filling with nan'.format(key))
            for bin_key in range(len(np.arange(self.hplevel_and_binning[key][0],self.hplevel_and_binning[key][1],self.hplevel_and_binning[key][2]))):
                if bin_key not in np.unique(df[key+'_']): df.loc[len(df),key+'_'] = bin_key
        dset_dr3 = xr.Dataset.from_dataframe(
            df.set_index([key + "_" for key in self.hplevel_and_binning.keys()])
        )
        dict_coords = {}
        for key in self.hplevel_and_binning.keys():
            if key == "healpix":
                continue
            dict_coords[key + "_"] = np.arange(
                self.hplevel_and_binning[key][0] + self.hplevel_and_binning[key][2] / 2, self.hplevel_and_binning[key][1], self.hplevel_and_binning[key][2]
            )
        dset_dr3 = dset_dr3.assign_coords(dict_coords)
        dset_dr3 = dset_dr3.rename({"healpix_": "ipix"})
        super().__init__(dset_dr3)


# Selection functions ported from gaiaverse's work ----------------
class EDR3RVSSelectionFunction(SelectionFunctionBase, fetch_utils.DownloadMixin):
    """Internal selection function for the RVS sample in EDR3.

    This has been ported from the selectionfunctions by Gaiaverse team.

    NOTE: The definition of the RVS sample is not the same as DR3RVSSelectionFunction.
    """

    __bibtex__ = """
@ARTICLE{2022MNRAS.509.6205E,
       author = {{Everall}, Andrew and {Boubert}, Douglas},
        title = "{Completeness of the Gaia verse - V. Astrometry and radial velocity sample selection functions in Gaia EDR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, methods: statistical, stars: statistics, Galaxy: kinematics and dynamics, Galaxy: stellar content, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {6205-6224},
          doi = {10.1093/mnras/stab3262},
archivePrefix = {arXiv},
       eprint = {2111.04127},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.6205E},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""

    datafiles = {
        "rvs_cogv.h5": "https://zenodo.org/record/8300616/files/rvs_cogv.h5"
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


class SubsampleSelectionFunctionHMLE():
    """Hierarchical maximum-likelihood estimate for the subsample selection function.

    This function gives the probability

        P(is in subsample| is in Gaia and has the certain HEALPix [, G [, color]])

    The Binomial MLE is used for estimation of the probability in the
    HEALPix [, G [, color]] bins. If a bin is empty (no trials), the upper
    HEALPix level will be used for estimate. If it is empty too, the next upper
    will be used, etc.

    Args:
    subsample_query : str, optional
    file_name : str, optional
        File name (without extension) to store the fetched data.
    hplevel_and_binning : dict, optional
    data : pandas.DataFrame or xarray.Dataset or dict, optional

    These are the possible use cases:
    1. `subsample_query`, `file_name` and `hplevel_and_binning` are given
       The data will be collected through the Gaia TAP+ interface then
        processed.
    2. No parameters are passed
       An empty class instance is created. The data should be provided later by
       the user and processed with the `use` method.
    3. An instance of the `SubsampleSelectionFunction` class is passed to the
       function `use`. This assumes that the data has already been collected.
    4. `pandas.DataFrame` and `hplevel_and_binning` are passed to the function
       `use`.
    5. `xarray.Dataset` and `hplevel_and_binning` are passed to the function
       `use`.
    """


    def __init__(self, subsample_query=None, file_name=None, hplevel_and_binning=None, z=None):
        if subsample_query and file_name and hplevel_and_binning:
            # Use case #1
            ssf = SubsampleSelectionFunction(subsample_query, file_name, hplevel_and_binning)
            self.use(ssf.ds, hplevel_and_binning, z)
            return
        if (subsample_query is None) and (file_name is None) and (hplevel_and_binning is None):
            # Use case #2, #3, ...
            return
        else:
            raise ValueError("`subsample_query`, `file_name` and `hplevel_and_binning` must be set simultaneously or not.")


    def use(self, obj, hplevel_and_binning=None, z=None):
        if type(obj) is SubsampleSelectionFunction:
            # Use case #3
            self.use_dataset(obj.ds, obj.hplevel_and_binning, z)
        elif type(obj) is xr.Dataset:
            # Use case #4
            self.use_dataset(obj, hplevel_and_binning, z)
        elif type(obj) is pd.DataFrame:
            # Use case #5
            self.use_pandas(obj, hplevel_and_binning, z)
        else:
            raise ValueError("The allowed types for `obj` are `SubsampleSelectionFunction`, `xarray.Dataset`, and `pandas.Dataframe`.")

        return self


    def use_dataset(self, ds, hplevel_and_binning, z=None):
        """Evaluate completeness using data collected in the native format at the `SubsampleSelectionFunction` class.

        Args:
        ds: xarray.Dataset
        hplevel_and_binning : dict
        z : float, optional
        """

        # Check dimensions
        assert ds['n'].shape == ds['k'].shape

        # Save it for future use in `evaluate`
        self.hplevel_and_binning = hplevel_and_binning

        # Convert types
        n = ds['n'].fillna(0).astype(int).to_numpy()
        k = ds['k'].fillna(0).astype(int).to_numpy()

        # Correct counts
        n[n < 0] = 0
        k[k < 0] = 0
        mask = k > n
        k[mask] = n[mask]

        if z is not None:
            nn, kk, pp, ci_lo, ci_hi = self.evaluate(n, k, z)
            self.finalize(nn, kk, pp, ci_lo, ci_hi)
        else:
            nn, kk, pp = self.evaluate(n, k, z)
            self.finalize(nn, kk, pp)


    def use_pandas(self, df, hplevel_and_binning, z=None):
        """Evaluate completeness using data collected in the 'melted' format.

        Args:
        df: pandas.DataFrame
        hplevel_and_binning : dict
        z : float, optional
        """

        columns = df.columns
        assert ('ipix' in columns) and ('n' in columns) and ('k' in columns), \
            "The data frame must contain 'ipix', 'n' and 'k' fields"

        # Save it for future use in `evaluate`
        self.hplevel_and_binning = hplevel_and_binning

        # No data means zero observations
        df[['n', 'k']].fillna(0, inplace=True)

        # Collect keys and dimensions
        keys = [ 'ipix' ]
        dims = [ hp.order2npix(self.hplevel_and_binning['healpix']) ]
        for key, val in self.hplevel_and_binning.items():
            if key == 'healpix':
                continue
            keys.append(key + '_')
            dims.append(len(np.arange(val[0], val[1], val[2])))

        # Allocate data cubes of integers
        # HEALPixels will occupy the zeroth dimension, the other dims will be
        # in the order as the keys appear in the `hplevel_and_binning` dict
        n = np.zeros(dims, dtype=int)
        k = n.copy()

        inds = tuple(df[k] for k in keys)
        n[inds] += df['n']
        k[inds] += df['k']

        # Correct counts
        n[n < 0] = 0
        k[k < 0] = 0
        mask = k > n
        k[mask] = n[mask]

        if z is not None:
            nn, kk, pp, ci_lo, ci_hi = self.evaluate(n, k, z)
            self.finalize(nn, kk, pp, ci_lo, ci_hi)
        else:
            nn, kk, pp = self.evaluate(n, k, z)
            self.finalize(nn, kk, pp)


    def evaluate(self, n, k, z=None):
        """Evaluate success probability and optionally confidence interval for every pixel and magnitude/color bin.

        Args:
        n : ndarray
        k : ndarray
        z : float, optional
        """

        hplevel = hp.npix2order(n.shape[0])

        #
        # Count bottom-up along the HEALPixels hierarchy

        nn = [n]
        kk = [k]

        npix = n.shape[0]
        ipix = np.arange(npix)
        for _ in range(hplevel, 0, -1):
            npix_ = npix // 4
            ipix_ = ipix.reshape((npix_, 4))

            n_ = nn[0][ipix_].sum(axis=1)
            k_ = kk[0][ipix_].sum(axis=1)

            nn.insert(0, n_)
            kk.insert(0, k_)

            npix = npix_
            ipix = np.arange(npix)

        #
        # Evaluate the hierarchical MLE

        pp = []
        ci_lo = []
        ci_hi = []

        if z is not None:
            z2 = z**2

        # Non-informative prior estimate at the global level
        nn00 = nn[0].sum(axis=0)
        kk00 = kk[0].sum(axis=0)
        p_ = (kk00 + 1.0) / (nn00 + 2.0) + np.zeros_like(nn[0])

        if z is not None:
            # Confidence interval
            var = z * np.sqrt(p_*(1.0 - p_)*nn00 + 0.25*z2)
            ci_lo_ = (p_*nn00 + 0.5*z2 - var) / (nn00 + z2) + np.zeros_like(nn[0])
            ci_hi_ = (p_*nn00 + 0.5*z2 + var) / (nn00 + z2) + np.zeros_like(nn[0])

        for l in range(0, hplevel+1):
            n_ = nn[l]
            k_ = kk[l]

            if l > 0:
                # Use a previous-level estimate by default
                ipix_ = np.arange(n_.shape[0]) // 4
                p_ = p_[ipix_].copy()

                if z is not None:
                    # Confidence interval
                    ci_lo_ = ci_lo_[ipix_].copy()
                    ci_hi_ = ci_hi_[ipix_].copy()

            # Get the MLE and CI where they're defined
            mask = n_ > 0
            p_[mask] = k_[mask] / n_[mask]
            pp.append(p_)

            if z is not None:
                # Confidence interval
                var = z * np.sqrt(p_[mask]*(1.0 - p_[mask])*n_[mask] + 0.25*z2)
                ci_lo_[mask] = (k_[mask] + 0.5*z2 - var) / (n_[mask] + z2)
                ci_hi_[mask] = (k_[mask] + 0.5*z2 + var) / (n_[mask] + z2)
                ci_lo.append(ci_lo_)
                ci_hi.append(ci_hi_)

        if z is not None:
            return nn, kk, pp, ci_lo, ci_hi
        else:
            return nn, kk, pp


    def finalize(self, nn, kk, pp, ci_lo=None, ci_hi=None):
        """Collect everything into a list of the datasets, one for each HEALPix level.
        """

        coords = OrderedDict()
        coords['ipix'] = None
        for key, val in self.hplevel_and_binning.items():
            if key == 'healpix':
                continue
            coords[key + '_'] = np.arange(val[0], val[1], val[2])
        #print("\n* finalize")
        #print("coords =", coords)

        self.hds = []
        #hplevel = self.hplevel_and_binning['healpix']
        #for l in range(0, hplevel+1):
        for l in range(len(pp)):
            coords['ipix'] = list(range(hp.order2npix(l)))

            ds = xr.DataArray(logit(pp[l]), name='logitp', coords=coords).to_dataset()
            ds['n'] = xr.DataArray(nn[l], name='n', coords=coords)
            ds['k'] = xr.DataArray(kk[l], name='k', coords=coords)

            if ci_lo is not None:
                ds['ci_lo'] = xr.DataArray(ci_lo[l], name='ci_lo', coords=coords)
            if ci_hi is not None:
                ds['ci_hi'] = xr.DataArray(ci_hi[l], name='ci_hi', coords=coords)
            self.hds.append(ds)

        return self.hds


    def query(self, coords, hplevel=-1, return_confidence=False, fill_nan=False, **kwargs):
        """Query the selection function at the given coordinates.

        Args:
        coords : astropy.coordinates.SkyCoord
            Sky coordinates as an astropy coordinates instance.
        hplevel : int, optional
            HEALPixel order. If omitted, the largest (finest) possible is used.
        return_confidence : bool, optional
            Whether to return the confidence interval (its lower and upper bounds).
        fill_nan : bool, optional
            There should not be any NaNs in the data. This parameter is left
            for backwards compatibility.
        kwargs : key-value pairs
            The key is a variable name (with '_' suffix), the value is an array
            of the values of the variable where to interpolate. The shape of
            the values must be the same as the shape of the 'coords'.

        Other factors that determine this selection function should be given
        as keyword arguments of the same shape as coords.

        Returns:
        numpy.ndarray
            Selection probabilities
        """

        ipix = utils.coord2healpix(coords, 'icrs', hp.order2nside(hplevel), nest=True)
        ipix = xr.DataArray(np.atleast_1d(ipix))

        d = {}
        factors = set(self.hds[hplevel]['logitp'].dims) - set({'ipix'})
        for k in factors:
            if k not in kwargs:
                raise ValueError(f"{k} values are missing.")
            d[k] = xr.DataArray(np.atleast_1d(kwargs[k]))
        d['method'] = 'nearest'
        d['kwargs'] = dict(fill_value=None)  # extrapolates

        out = self.hds[hplevel]['logitp'].interp(ipix=ipix, **d).to_numpy()
        if len(coords.shape) == 0:
            out = out.squeeze()

        if return_confidence:
            out_ci_lo = self.hds[hplevel]['ci_lo'].interp(ipix=ipix, **d).to_numpy()
            out_ci_lo.squeeze()
            out_ci_hi = self.hds[hplevel]['ci_hi'].interp(ipix=ipix, **d).to_numpy()
            out_ci_hi.squeeze()
            return expit(out), out_ci_lo, out_ci_hi
        else:
            return expit(out)
