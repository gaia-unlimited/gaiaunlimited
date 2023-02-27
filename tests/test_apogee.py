from gaiaunlimited.selectionfunctions.apogee import apogee_sf
from astropy.coordinates import SkyCoord

def test_apogee():
    c1 = SkyCoord(226,60,frame='galactic',unit='deg')
    c2 = SkyCoord(228,60,frame='galactic',unit='deg')
    c3 = SkyCoord(224,60,frame='galactic',unit='deg')
    assert apogee_sf(11,1,c3) == 0
    assert apogee_sf(11,1,c1) > 0
    assert apogee_sf(11,1,c2) > 0
    assert apogee_sf(11,1,c2) > apogee_sf(11,1,c1)
