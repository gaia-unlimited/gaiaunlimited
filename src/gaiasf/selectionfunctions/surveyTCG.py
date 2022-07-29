import numpy as np
import healpy as hp
from gaiasf import fetch_utils, utils
import h5py


class DR3SelectionFunctionTCG:
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
        pointIndices = np.array(
            [np.where(ipix_map == foo)[0][0] for foo in ipix]
        )  # horrendous but works
        allM10 = m10_map[pointIndices]
        c = selectionFunction(gmag.astype(float), allM10)
        return c

    @classmethod
    def from_patch(cls, *args, **kwargs):
        pass
        # make m10map
        # return cls(m10map)

    def to_hdf5(self, filename):
        pass

    @classmethod
    def from_file(cls, filename):
        pass
        # TODO
        # return cls(m10map)


class DR3SelectionFunctionTCG7(DR3SelectionFunctionTCG, fetch_utils.DownloadMixin):
    datafiles = {
        "allsky_M10_hpx7.hdf5": "https://github.com/TristanCantatGaudin/GaiaCompleteness/blob/main/allsky_M10_hpx7.hdf5?raw=true"
    }

    def __init__(self):
        with h5py.File(self._get_data("allsky_M10_hpx7.hdf5"), "r") as f:
            m10_order7 = f["data"][()]
        super().__init__(m10_order7)


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


def selectionFunction(G, m10):
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
    a, b, c, d, e, f, x, y, z, lim = dict(
        a=1.0154179774831278,
        b=-0.008254847738351057,
        c=0.6981959151433699,
        d=-0.07503539255843136,
        e=1.7491113533977052,
        f=0.4541796235976577,
        x=-0.06817682843336803,
        y=1.5712714454917935,
        z=-0.12236281756914291,
        lim=20.53035927443456,
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
    predictedG0 = a * m10 + b
    predictedG0[m10 > lim] = c * m10[m10 > lim] + (a - c) * lim + b
    #
    predictedInvslope = x * m10 + y
    predictedInvslope[m10 > lim] = z * m10[m10 > lim] + (x - z) * lim + y
    #
    predictedShape = d * m10 + e
    predictedShape[m10 > lim] = f * m10[m10 > lim] + (d - f) * lim + e

    if m10wasFloat and isinstance(G, (int, float)) == True:
        return mask * sigmoid(G, predictedG0, predictedInvslope, predictedShape)[0]
    else:
        return mask * sigmoid(G, predictedG0, predictedInvslope, predictedShape)
