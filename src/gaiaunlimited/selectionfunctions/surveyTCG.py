import numpy as np
import healpy as hp
import h5py
from astroquery.gaia import Gaia
from astropy.table import Table
import astropy_healpix as ah
import astropy.units as u
from astropy.coordinates import SkyCoord

from .. import fetch_utils, utils

__all__ = [
    "DR3SelectionFunctionTCG",
    "build_patch_map",
    "sigmoid",
    "m10_to_completeness",
]

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1  # default is 50 rows max, -1 for unlimited


class DR3SelectionFunctionTCG(fetch_utils.DownloadMixin):
    """Model of the Gaia DR3 survey selection function calibrated on DECaPS.

    Available in three flavours:

    *   mode = 'hpx7' (default)
        Uses a map precomputed in healpix regions of order 7.

    *   mode='multi'
        Uses a precomputed map of variable resolution,
        with healpixels as small as order 10, provided they contain at least 20 sources.

    *   mode='patch'
        The field of view is a circular patch of radius 'radius' centered on (ra,dec).
        The spatial resolution will vary across the field of view,
        from healpix order 6 to 12, enforcing that bins must contain
        at least min_points sources (default 20). A low number makes the map
        more detailed but also noisier.

    Arguments:
        mode: 'hpx7' or 'multi' or 'patch' (defaults to 'hpx7')

    In 'patch' mode only:
        ra: right ascension of centre of field of view, in degrees
        dec: declination of centre of field of view, in degrees
        size: width/height of the square field of view, in degrees
        min_points: minimum number of sources used to compute the map

    If you use this model in a publication please cite::

        @ARTICLE{2023A&A...669A..55C,
               author = {{Cantat-Gaudin}, Tristan and {Fouesneau}, Morgan and {Rix}, Hans-Walter and {Brown}, Anthony G.~A. and {Castro-Ginard}, Alfred and {Kostrzewa-Rutkowska}, Zuzanna and {Drimmel}, Ronald and {Hogg}, David W. and {Casey}, Andrew R. and {Khanna}, Shourya and {Oh}, Semyeong and {Price-Whelan}, Adrian M. and {Belokurov}, Vasily and {Saydjari}, Andrew K. and {Green}, G.},
                title = "{An empirical model of the Gaia DR3 selection function}",
              journal = {\aap},
             keywords = {astrometry, catalogs, methods: data analysis, methods: statistical, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
                 year = 2023,
                month = jan,
               volume = {669},
                  eid = {A55},
                pages = {A55},
                  doi = {10.1051/0004-6361/202244784},
        archivePrefix = {arXiv},
               eprint = {2208.09335},
         primaryClass = {astro-ph.GA},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...669A..55C},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
    """

    datafiles = {
        "allsky_M10_hpx7.hdf5": "https://zenodo.org/record/8063930/files/allsky_M10_hpx7.hdf5",
        "allsky_uniq_10.fits": "https://zenodo.org/record/8063930/files/allsky_uniq_10.fits",
    }

    def __init__(self, mode="hpx7", coord=None, radius=None, min_points=20):
        self.mode = mode

        if self.mode == "multi":
            self.m10map = Table.read(self._get_data("allsky_uniq_10.fits"))
            # For every healpixel large or small, represent it with is first order 29 division.
            # Once ordered, any sky position can quickly be linked to its nearest healpixel
            # (which will correspond to the one containing it)
            max_level = 29
            level, ipix = ah.uniq_to_level_ipix(self.m10map["UNIQ"])
            self.index = ipix * (2 ** (max_level - level)) ** 2
            self.sorter = np.argsort(self.index)
            self.max_nside = ah.level_to_nside(max_level)

        elif self.mode == "patch":
            self.coord = coord
            self.radius_patch = radius
            self.min_points = min_points
            # The m10map is not read from a file, we have to build it on the fly:
            self.m10map = build_patch_map(coord, radius, min_points)

        else:
            # Any invalid choice of mode defaults to the healpix 7 map.
            if mode != "hpx7":
                print("Mode %s unknown, defaulting to 'hpx7'." % (mode))
            self.mode = "hpx7"
            with h5py.File(self._get_data("allsky_M10_hpx7.hdf5"), "r") as f:
                self.m10map = f["data"][()]

    def query(self, coords, gmag):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            gmag (float or array): G magnitudes. Should have the same shape as coords.

        Returns:
            prob: array of selection probabilities.
        """
        if not isinstance(coords, SkyCoord):
            print(
                "*** WARNING: The coordinates do not seem to be an astropy.coord.SkyCoord object."
            )
            print(
                "This could lead to the error:\n     \"object has no attribute 'frame'\""
            )
            print(
                "The syntax to query the completeness map is:\n    mapName.query( coordinates , gmags )"
            )
        if self.mode == "multi":
            ra = coords.ra
            dec = coords.dec
            # Determine the NESTED pixel index of the target sky location at that max resolution.
            match_ipix = ah.lonlat_to_healpix(ra, dec, self.max_nside, order="nested")
            # Do a binary search for that value:
            i = self.sorter[
                np.searchsorted(
                    self.index, match_ipix, side="right", sorter=self.sorter
                )
                - 1
            ]
            # That pixel contains the target sky position.
            allM10 = self.m10map[i]["M10"]
            prob = m10_to_completeness(gmag.astype(float), allM10)
            return prob
        else:
            if coords.shape != np.shape(gmag):
                raise ValueError(
                    f"Input shape mismatch: {coords.shape} != {gmag.shape}"
                )
            order_map = self.m10map[:, 0].astype(np.int64)
            ipix_map = self.m10map[:, 1].astype(np.int64)
            m10_map = self.m10map[:, 2]
            nside = 2 ** order_map[0]
            ipix = utils.coord2healpix(coords, "icrs", nside)
            # if using custom maps, the user might query a point outside the map:
            is_in_map = np.in1d(ipix, ipix_map)
            if np.all(is_in_map) == False:
                print("Warning: some requested points are outside the map.")
                # print(coords[~is_in_map])
                # find the missing ipix, temporarily add them with value Nan
                missingIpix = sorted(set(ipix[~is_in_map]))
                for mip in missingIpix:
                    ipix_map = np.append(ipix_map, mip)
                    m10_map = np.append(m10_map, np.nan)
            pointIndices = np.array(
                [np.where(ipix_map == foo)[0][0] for foo in ipix]
            )  # horrendous but works, could be clearer with np.in1d?
            allM10 = m10_map[pointIndices]
            prob = m10_to_completeness(gmag.astype(float), allM10)
            return prob


def build_patch_map(coord, radius, min_points=20):
    """
    Query the Gaia database and create a high-resolution healpix map of the M_10
    parameter for a given circular patch of sky.
    The pixels without a sufficient number of sources will be grouped together.

    Args:
        coord (:obj:`astropy.coordinates.SkyCoord`): sky coordinates of the center of the patch, as an astropy SkyCoord object
        radius (float): the radius of the patch, in degrees
        min_points (int): minimum number of sources used to compute M_10 in a given pixel.
            A given region will be subdivided into four higher-order regions
            if all its subdivisions contain more than min_points points.

    Returns:
        A numpy array of shape (3,N) where the first column is the order of the maximum resolution reached,
        the second column is the healpixel number, and the third is the M_10 value in that healpixel.
    """
    ra_patch = coord.icrs.ra / u.degree
    dec_patch = coord.icrs.dec / u.degree
    print("Querying the Gaia archive...")
    queryStringGaia = """SELECT ra, dec, source_id,phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT(ra,dec),CIRCLE(%f, %f, %f))
    and astrometric_matched_transits<11
    and phot_g_mean_mag<50""" % (
        ra_patch,
        dec_patch,
        radius,
    )
    print(queryStringGaia)
    job = Gaia.launch_job_async(queryStringGaia)
    GaiaT = job.get_results()
    print(f"Obtained {len(GaiaT)} sources.")
    # - - - TEMPORARY FIX - - - 
    # in astroquery v0.4.7 (March 2024) some columns names
    # are now passed in upper case (see issues 2911 and 2965).
    try:
        GaiaT["source_id"] = GaiaT["SOURCE_ID"]
    except:
        pass

    # find all the potential hpx ids of queried sources:
    allHpx6 = sorted(set(GaiaT["source_id"] // (2**35 * 4 ** (12 - 6))))
    allHpx7 = [foo for i in allHpx6 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]]
    allHpx8 = [foo for i in allHpx7 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]]
    allHpx9 = [foo for i in allHpx8 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]]
    allHpx10 = [
        foo for i in allHpx9 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
    ]
    allHpx11 = [
        foo for i in allHpx10 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
    ]
    allHpx12 = [
        foo for i in allHpx11 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
    ]
    # identify which ones have centers inside the user-requested patch:
    allGoodHpx12 = []
    for h in allHpx12:
        rah, dech = hp.pix2ang(2**12, h, nest=True, lonlat=True)
        if coord.separation(SkyCoord(ra=rah, dec=dech, unit="deg")) < u.Quantity(
            radius, u.deg
        ):
            allGoodHpx12.append(h)

    fineMap = [np.nan for foo in allGoodHpx12]
    for stepUp in range(5):
        print(f"Grouping the stars by hpx level {12-stepUp}...")
        sourceHpxThisOrder = np.array(GaiaT["source_id"]) // (
            2**35 * 4 ** (12 - (12 - stepUp))
        )
        for i, h in enumerate(allGoodHpx12):
            if np.isnan(fineMap[i]):
                gI = GaiaT["phot_g_mean_mag"][sourceHpxThisOrder == h // 4**stepUp]
                # print(i,h,gI); input()
                if len(gI) >= min_points:
                    fineMap[i] = np.ma.median(gI)
                else:
                    pass
    print("Done.")
    fineMap = np.array(fineMap)
    allGoodHpx12 = np.array(allGoodHpx12)
    order = 12 * np.ones_like(allGoodHpx12)
    m10_map = np.column_stack((order, allGoodHpx12, fineMap))
    return m10_map


def sigmoid(G, G0, invslope, shape):
    """Generalized sigmoid function.

    Note: this function is not robust to numerical issues but works within the range of values we feed it.

    Args:
        G (nd.array): where to evaluate the function
        G0 (float): inflection point
        invslope (float): steepness of the linear part. Shallower for larger values
        shape (float): if shape=1, model is the classical logistic function,
            shape converges to zero, then the model is a Gompertz function.

    Returns:
        evaluation of the model f(G) = 1 - (0.5 * (np.tanh(delta / invslope) + 1)) ** shape
    """
    delta = G - G0
    return 1 - (0.5 * (np.tanh(delta / invslope) + 1)) ** shape


def m10_to_completeness(G, m10):
    """Predicts the completeness at magnitude G, given a value of M_10 read from a precomputed map.

    Args:
        G (float or nd.array): where to evaluate the function
        m10 (float or nd.array): the value of M_10 in a given region

    Returns:
        sf(G) between 0 and 1.
        The shape of the output will match the input:
            * if given an array (i.e. an array of positions) the output is an array
            * if given an array of Gmag and either one position or a matching array of positions, the output is also an array
            * if only given scalars, the output is one number.

    """
    # These are the best-fit value of the free parameters we optimised in our model:
    ax, bx, cx, ay, by, cy, az, bz, cz, lim = dict(
        ax=0.9848761394197864,
        bx=0.6473155510230146,
        cx=0.6929084598209412,
        ay=-0.003935382139847386,
        by=0.2230529402297744,
        cy=-0.09331877468160235,
        az=0.006144107896473064,
        bz=0.03681705933744438,
        cz=0.35140564525722895,
        lim=20.519369625540833,
    ).values()

    if isinstance(m10, float):
        m10 = np.array([m10])
        m10wasFloat = True
    else:
        m10 = np.array(m10)
        m10wasFloat = False

    # if there are NaNs in the list (e.g. a map with holes in it) we will get a RuntimeWarning,
    # so we replace NaNs with zeros, and will put NaNs back at the end:
    mask = m10 / m10
    m10 = np.nan_to_num(m10)
    #
    predictedG0 = ax * m10 + bx
    predictedG0[m10 > lim] = cx * m10[m10 > lim] + (ax - cx) * lim + bx
    #
    predictedInvslope = ay * m10 + by
    predictedInvslope[m10 > lim] = cy * m10[m10 > lim] + (ay - cy) * lim + by
    #
    predictedShape = az * m10 + bz
    predictedShape[m10 > lim] = cz * m10[m10 > lim] + (az - cz) * lim + bz

    if m10wasFloat and isinstance(G, (int, float)) == True:
        return mask * sigmoid(G, predictedG0, predictedInvslope, predictedShape)[0]
    else:
        return mask * sigmoid(G, predictedG0, predictedInvslope, predictedShape)
