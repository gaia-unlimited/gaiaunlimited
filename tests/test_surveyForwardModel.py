import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from gaiaunlimited.selectionfunctions import DR3SelectionFunctionForwardModel
from gaiaunlimited.utils import get_healpix_centers


@pytest.fixture(scope="module")
def sf():
    return DR3SelectionFunctionForwardModel()


def test_table_shapes(sf):
    npix = hp.order2npix(7)
    assert sf.nside == 128
    assert sf.p.shape == (npix, len(sf.g_centers))
    assert sf.N_eff.shape == (npix,)
    assert np.allclose(sf.g_edges[[0, -1]], [1.0, 22.0])
    assert np.nanmin(sf.p) >= 0 and np.nanmax(sf.p) <= 1


def test_scalar_query(sf):
    c = SkyCoord(ra=10.0, dec=-10.0, unit="deg", frame="icrs")
    result = sf.query(c, 20.0)
    assert result.shape == ()
    assert 0 <= result <= 1


def test_array_query_matches_table(sf):
    # independent lookup: nested HEALPix order 7 in ICRS, 0.2 mag bins from G=1
    ra = np.array([10.0, 200.0, 266.4, 80.0])
    dec = np.array([-10.0, 40.0, -29.0, -69.0])
    gmag = np.array([17.05, 19.5, 20.9, 21.0])
    ipix = hp.ang2pix(128, ra, dec, nest=True, lonlat=True)
    igbin = np.ceil((gmag - 1.0) / 0.2).astype(int) - 1  # bins are (lo, hi]
    result = sf.query(SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"), gmag)
    assert result.shape == (4,)
    assert np.array_equal(result, sf.p[ipix, igbin], equal_nan=True)


def test_out_of_range_is_nan(sf):
    coords = get_healpix_centers(0)
    gmag = np.full(coords.shape, 23.0)
    assert np.all(np.isnan(sf.query(coords, gmag)))
    gmag = np.full(coords.shape, 0.5)
    assert np.all(np.isnan(sf.query(coords, gmag)))


def test_galactic_frame_input(sf):
    c_icrs = SkyCoord(ra=[266.4], dec=[-29.0], unit="deg", frame="icrs")
    assert np.array_equal(
        sf.query(c_icrs, np.array([20.5])),
        sf.query(c_icrs.galactic, np.array([20.5])),
        equal_nan=True,
    )


def test_shape_mismatch(sf):
    coords = get_healpix_centers(0)
    with pytest.raises(ValueError):
        sf.query(coords, np.ones(3))


def test_effective_scanning_law(sf):
    coords = get_healpix_centers(0)
    out = sf.query_effective_scanning_law(coords)
    assert set(out) == {"N_eff", "N_usable", "N_error"}
    for v in out.values():
        assert v.shape == coords.shape
    assert np.all(out["N_eff"] > 0)


def test_bin_edges_match_paper(sf):
    # Listing 2 of the paper uses G = 19.5, 20.1, 20.7, 21 for the bins
    # (19.4, 19.6), (20.0, 20.2), (20.6, 20.8), (20.8, 21.0)
    for g, lo in [(19.5, 19.4), (20.1, 20.0), (20.7, 20.6), (21.0, 20.8), (20.8, 20.6), (1.0, 1.0), (22.0, 21.8)]:
        i = np.searchsorted(sf.g_edges, g, side="left") - 1
        i = max(i, 0)
        assert np.isclose(sf.g_edges[i], lo), (g, sf.g_edges[i])
        c = SkyCoord(ra=10.0, dec=-10.0, unit="deg", frame="icrs")
        ipix = hp.ang2pix(128, 10.0, -10.0, nest=True, lonlat=True)
        assert np.array_equal(sf.query(c, g), sf.p[ipix, i], equal_nan=True)


def test_paper_listings_run(sf):
    # Listings 1 and 2 of Castro-Ginard et al. (2026), without plotting
    coords_of_centers = get_healpix_centers(7)
    scanning_law = sf.query_effective_scanning_law(coords_of_centers)
    assert scanning_law["N_eff"].shape == coords_of_centers.shape
    for g in np.array([19.5, 20.1, 20.7, 21]):
        prob_map = sf.query(coords_of_centers, np.full(len(coords_of_centers), g))
        assert prob_map.shape == coords_of_centers.shape
