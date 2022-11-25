import numpy as np
import healpy as hp
from gaiaunlimited import fetch_utils, utils
import h5py
from astroquery.gaia import Gaia
from astropy.table import Table
import astropy_healpix as ah 

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1  # default is 50 rows max, -1 for unlimited




class DR3SelectionFunctionTCG_multi(fetch_utils.DownloadMixin):
    """Model of the Gaia DR3 survey selection function calibrated on DECaPS.
       This implementation uses a precomputed map of variable resolution,
       with healpixels as small as order 10, provided they contain at least 20 sources."""

    datafiles = {
        "allsky_uniq_10.fits": "https://keeper.mpdl.mpg.de/f/c4fb744aa7e941928da6/?dl=1"
    }

    def __init__(self):
        self.skymap = Table.read(   self._get_data("allsky_uniq_10.fits")  )
        #For every healpixel large or small, represent it with is first order 29 division.
        #Once ordered, any sky position can quickly be linked to its nearest healpixel
        #(which will correspond to the one containing it)
        max_level = 29
        level, ipix = ah.uniq_to_level_ipix(self.skymap['UNIQ'])
        self.index = ipix * (2**(max_level - level))**2
        self.sorter = np.argsort(self.index)
        self.max_nside = ah.level_to_nside(max_level)

    def query(self, coords, gmag):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            gmag (float or array): G magnitudes. Should have the same shape as coords.

        Returns:
            prob: array of selection probabilities.
        """
        ra = coords.ra
        dec = coords.dec
        #Determine the NESTED pixel index of the target sky location at that max resolution.
        match_ipix = ah.lonlat_to_healpix(ra, dec, self.max_nside, order='nested')
        #Do a binary search for that value:
        i = self.sorter[np.searchsorted(self.index, match_ipix, side='right', sorter=self.sorter) - 1]
        #That pixel contains the target sky position.
        allM10 = self.skymap[i]['M10']
        prob = m10_to_completeness(gmag.astype(float), allM10)
        return prob


class DR3SelectionFunctionTCG:
    """Model of the Gaia DR3 survey selection function calibrated on DECaPS."""

    def __init__(self, m10map):
        # NOTE: HEAL pixelisation is in ICRS in these files!
        self.m10map = m10map

    def query(self, coords, gmag):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            gmag (float or array): G magnitudes. Should have the same shape as coords.

        Returns:
            prob: array of selection probabilities.
        """
        if coords.shape != np.shape(gmag):
            raise ValueError(f"Input shape mismatch: {coords.shape} != {gmag.shape}")
        order_map = self.m10map[:, 0].astype(np.int64)
        ipix_map = self.m10map[:, 1].astype(np.int64)
        m10_map = self.m10map[:, 2]
        nside = 2 ** order_map[0]
        ipix = utils.coord2healpix(coords, "icrs", nside)
        # if using custom maps, the user might query a point outside the map:
        is_in_map = np.in1d(ipix, ipix_map)
        if np.all(is_in_map) == False:
            print("Warning: the following points are outside the map:")
            print(coords[~is_in_map])
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

    # @classmethod
    # def from_patch(cls, *args, **kwargs):
    #    pass
    #    # make m10map
    #    # return cls(m10map)


class DR3SelectionFunctionTCG_hpx7(DR3SelectionFunctionTCG, fetch_utils.DownloadMixin):
    """Initialises the model from the all-sky map precomputed in healpix order 7 (Nside=128)."""

    datafiles = {
        "allsky_M10_hpx7.hdf5": "https://github.com/TristanCantatGaudin/GaiaCompleteness/blob/main/allsky_M10_hpx7.hdf5?raw=true"
    }

    def __init__(self):
        with h5py.File(self._get_data("allsky_M10_hpx7.hdf5"), "r") as f:
            m10_order7 = f["data"][()]
        super().__init__(m10_order7)


class DR3SelectionFunctionTCG_from_patch(DR3SelectionFunctionTCG):
    """Initialises the model for a requested patch of sky.
    The field of view is a square of width 'size' centered on (ra,dec).
    The spatial resolution will vary across the field of view,
    from healpix order 6 to 12, enforcing that bins must contain
    at least min_points sources (default 5). A low number makes the map
    more detailed but also noisier.

    Args:
        ra: right ascension of centre of field of view, in degrees
        dec: declination of centre of field of view, in degrees
        size: width/height of the field of view, in degrees
        min_points: minimum number of sources used to compute the map"""

    def __init__(self, ra, dec, size, min_points=20):
        self.ra = ra
        self.dec = dec
        self.size = size
        self.min_points = min_points

        scale = 1.0 / np.cos(np.radians(self.dec))

        queryStringGaia = """SELECT ra, dec, source_id,phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE ra>%.3f and ra<%.3f
        and dec>%.3f and dec<%.3f
        and astrometric_matched_transits<11
        and phot_g_mean_mag<50""" % (
            self.ra - scale * self.size / 2,
            self.ra + scale * self.size / 2,
            self.dec - self.size / 2,
            self.dec + self.size / 2,
        )
        print("Querying the Gaia archive...")
        job = Gaia.launch_job_async(queryStringGaia)
        GaiaT = job.get_results()
        print(f"{len(GaiaT)} sources downloaded.")

        # find all the potential hpx ids of queried sources:
        allHpx6 = sorted(set(GaiaT["source_id"] // (2**35 * 4 ** (12 - 6))))
        allHpx7 = [
            foo for i in allHpx6 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        ]
        allHpx8 = [
            foo for i in allHpx7 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        ]
        allHpx9 = [
            foo for i in allHpx8 for foo in [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        ]
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
            # print(rah,dech)
            if (
                rah < self.ra + scale * self.size / 2
                and rah > self.ra - scale * size / 2
                and dech < self.dec + size / 2
                and dech > self.dec - size / 2
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
                    gI = GaiaT["phot_g_mean_mag"][
                        sourceHpxThisOrder == h // 4**stepUp
                    ]
                    # print(i,h,gI); input()
                    if len(gI) >= min_points:
                        fineMap[i] = np.median(gI)
                    else:
                        pass
        print("Done.")
        fineMap = np.array(fineMap)
        allGoodHpx12 = np.array(allGoodHpx12)
        order = 12 * np.ones_like(allGoodHpx12)
        m10_map = np.column_stack((order, allGoodHpx12, fineMap))
        self.allGoodHpx12 = allGoodHpx12
        self.fineMap = fineMap
        super().__init__(m10_map)

    def display(self, G=None):
        """Uses healpy.gnomview to display the map."""
        # fill the fullsky map
        npix = hp.nside2npix(2**12)
        hpx_map = np.zeros(npix, dtype=float) * np.nan
        idx = self.allGoodHpx12
        if G is None:
            cmap = "turbo"
            hpx_map[idx] = self.fineMap
        else:
            cmap = "viridis"
            hpx_map[idx] = m10_to_completeness(G, self.fineMap)
        hp.gnomview(
            hpx_map,
            rot=[self.ra, self.dec],
            nest=True,
            xsize=2.1 * 60 * self.size,
            reso=0.5,
            cmap=cmap,
        )
        import matplotlib.pyplot as plt

        plt.show()


def sigmoid(
    G: np.ndarray, G0: np.ndarray, invslope: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """Generalized sigmoid function

    Parameters
    ----------
    G: nd.array
        where to evaluate the function
    G0: float
        inflection point
    invslope: float
        steepness of the linear part. Shallower for larger values
    shape: float
        if shape=1, model is the classical logistic function,
        shape converges to zero, then the model is a Gompertz function.

    Returns
    -------
    f(G) evaluation of the model.

        FIXME: this function is not robust to numerical issues (but works within the range of values we feed it)
    """
    delta = G - G0
    return 1 - (0.5 * (np.tanh(delta / invslope) + 1)) ** shape


def m10_to_completeness(G, m10):
    """Predicts the completeness at magnitude G, given a value of M_10 read from a precomputed map.

    Parameters
    ----------
    G:   float or nd.array
                    where to evaluate the function
    m10: float or nd.array
                    the value of M_10 in a given region

    Returns
    -------
    sf(G) between 0 and 1.
    The shape of the output will match the input:
            if given an array (i.e. an array of positions) the output is an array
            if given an array of Gmag and either one position or a matching array of positions, the output is also an array
            if only given scalars, the output is one number.

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
