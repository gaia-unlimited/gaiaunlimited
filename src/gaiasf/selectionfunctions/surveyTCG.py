import numpy as np
import pandas as pd
import healpy as hp
from gaiasf import fetch_utils
import h5py


class DR3SelectionFunctionTCG:

    modelparams = dict(
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
    )

    def __init__(self, m10map):
        self._get_data("allsky_m10_hpx7.h5")

    def query(
        self,
    ):
        pass

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
    datafiles = {"allsky_m10_hpx7.h5": "TODO"}

    def __init__(self):
        with h5py.File(self._getdata("allsky_m10_hpx7.h5")) as f:
            m10_order7 = f["data"][()]
        super().__init__(m10_order7)


def sigmoid(G: np.ndarray, G0: np.ndarray, invslope: float, shape: float) -> np.ndarray:
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
    a, b, c, d, e, f, x, y, z, lim = get_model_parameters().values()

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


def selectionFunctionRADEC(ra, dec, G, dfm10) -> np.ndarray:
    """Evaluate the completeness at ra, dec, for given G array.

    Parameters
    ----------
    ra : float or nd.array
            right ascension in degrees
    dec: float or nd.array
            declination in degrees
    G : float np.ndarray
            which magnitude to evaluate the completeness
    dfm10: np.array or pd.DataFrame
            the M10 array

    Returns
    -------
    Evaluated completeness

    TO DO:
            Check the format of the provided map. Currently an incorrect format can lead to a variety of error messages.
    """
    if len(dfm10) == 3:
        MAPm10, xedges, yedges = dfm10
        if (
            ra < min(xedges)
            or ra > max(xedges)
            or dec < min(yedges)
            or dec > max(yedges)
        ):
            print(
                "\x1b[6;30;41m"
                + "Problem!"
                + "\x1b[0m"
                + f" position {ra} {dec} outside map coverage"
            )
            print(
                f"map coverage: ra {min(xedges)} {max(xedges)}, dec {min(yedges)} {max(yedges)}"
            )
            print("in function selectionFunctionRADEC")
            return np.nan
        indRa = int((ra - xedges[0]) / (xedges[1] - xedges[0]))
        indDec = int((dec - yedges[0]) / (yedges[1] - yedges[0]))
        m10 = MAPm10[indRa][indDec]
    else:
        # first identify the healpix order from the number of hpx in the map:
        nbHpx = len(dfm10)
        healpix_level = int(np.log(nbHpx / 12) / np.log(4))
        # then find the m10 value in the hpx corresponding to (ra,dec):
        m10 = dfm10[hp.ang2pix(2**healpix_level, ra, dec, lonlat=True, nest=True)]
    return selectionFunction(G, m10)
