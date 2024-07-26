from gaiaunlimited.scanninglaw import GaiaScanningLaw
from gaiaunlimited.utils import get_healpix_centers


def test_scanninglaw():
    try:
        sl = GaiaScanningLaw()
        cc = get_healpix_centers(0)
        t1, t2 = sl.query(cc.ra[0].deg, cc.dec[0].deg)
    except:
        assert False, "Scanning law query failed."
