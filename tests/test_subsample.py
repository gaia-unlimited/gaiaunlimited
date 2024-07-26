import numpy as np
import astropy.coordinates as coord
import astropy.units as u

from gaiaunlimited.selectionfunctions.subsample import (
    EDR3RVSSelectionFunction,
    DR3RVSSelectionFunction,
)
from gaiaunlimited.utils import get_healpix_centers


def test_subsample_sfs():
    for csf in [EDR3RVSSelectionFunction, DR3RVSSelectionFunction]:
        sf = csf()
        testargs_scalar = dict(
            coords=coord.SkyCoord(42.123 * u.deg, 0.23 * u.deg), g=12.1, c=0.8
        )
        result_scalar = sf.query(**testargs_scalar)
        assert np.ndim(result_scalar) == 0, "Scalar query did not return scalar."
        assert (
            result_scalar >= 0 and result_scalar <= 1.0
        ), "Selection probability is not between 0 and 1."

        cc = get_healpix_centers(5)
        gval = 12.1
        cval = 0.8
        testargs_1d = dict(
            coords=cc,
            g=np.ones_like(cc, dtype=float) * gval,
            c=np.ones_like(cc, dtype=float) * cval,
        )
        result_1d = sf.query(**testargs_1d)
        assert np.ndim(result_1d) == 1, "1d query did not return a 1d array."
        assert ((result_1d >= 0) | np.isnan(result_1d)).all() and (
            (result_1d <= 1.0) | np.isnan(result_1d)
        ).all(), "Selection probability is not between 0 and 1."
