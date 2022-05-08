import numpy as np
from .core import find_nearest, GaiaScanningLaw


def test_find_nearest():
    assert find_nearest([1, 2, 3, 4, 5], -5) == 0
    assert find_nearest([1, 2, 3, 4, 5], 3.15) == 2
    np.testing.assert_equal(
        find_nearest([1, 2, 3, 4, 5], [-5, 10, 0, 2, -1]), [0, 4, 0, 1, 0]
    )

