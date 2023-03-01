from gaiaunlimited.selectionfunctions.surveyTCG import DR3SelectionFunctionTCG_hpx7
from astropy.coordinates import SkyCoord
from gaiaunlimited.selectionfunctions.surveyTCG import DR3SelectionFunction
import numpy as np


def test_tcg():
    try:
        DR3SelectionFunctionTCG_hpx7()
    except:
        assert False, "Failed"


def test_multi():
    mapMulti = DR3SelectionFunction("multi")
    c_0_0 = mapMulti.query(
        SkyCoord(ra=[0], dec=[0], unit="deg", frame="icrs"), np.array([21])
    )
    c_ngc5139 = mapMulti.query(
        SkyCoord(ra=[201.8], dec=[-47.5], unit="deg", frame="icrs"), np.array([21])
    )
    assert c_0_0 > c_ngc5139


def test_hpx7():
    mapHpx7 = DR3SelectionFunction("hpx7")
    c_0_0 = mapHpx7.query(
        SkyCoord(ra=[0], dec=[0], unit="deg", frame="icrs"), np.array([21])
    )
    c_ngc5139 = mapHpx7.query(
        SkyCoord(ra=[201.8], dec=[-47.5], unit="deg", frame="icrs"), np.array([21])
    )
    assert c_0_0 > c_ngc5139


def test_patch():
    coord_patch = SkyCoord(ra=200, dec=-47, unit="deg", frame="icrs")
    mapPatch = DR3SelectionFunction("patch", coord_patch, radius=0.3, min_points=20)
    c_200_m47 = mapPatch.query(
        SkyCoord(ra=[200], dec=[-47], unit="deg", frame="icrs"), np.array([21])
    )
    assert c_200_m47 > 0
