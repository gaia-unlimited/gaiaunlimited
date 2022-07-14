import numpy as np

from gaiasf.selectionfunctions.survey import DR2SelectionFunction
from gaiasf.utils import get_healpix_centers


def test_dr2sf():
    x = DR2SelectionFunction()
    test_coords = get_healpix_centers(0)
    gmag = np.ones_like(test_coords) * 21.0

    # Compared values are from gaiaverse/selectionfunctions.
    # They can be slighly different because they do spline interpolation
    # but we do nearest.
    result_T = x.query(test_coords, gmag, use_modelT=True)
    ans_T = np.array(
        [
            0.50076593,
            0.98004925,
            0.99908801,
            0.8984253,
            0.99835906,
            0.55003379,
            0.99908801,
            0.56690795,
            0.99320507,
            0.96170959,
            0.57343536,
            0.98232239,
        ]
    )
    assert np.allclose(result, ans, atol=0.05)

    result = x.query(test_coords, gmag)
    ans = np.array(
        [
            0.47785934,
            0.94768624,
            0.99124919,
            0.82955219,
            0.98794938,
            0.51827671,
            0.99124919,
            0.52485844,
            0.97286789,
            0.90909688,
            0.53422559,
            0.95125774,
        ]
    )
    assert np.allclose(result, ans, atol=0.05)
