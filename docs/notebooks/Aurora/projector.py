
from tqdm import tqdm

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
from astropy_healpix import HEALPix



class Projector():

    def __init__(self):
        pass


    def make(self, dom, verbose=True):
        """
        Parameters
        ----------
        dom : Domain()
            Instance of Domain class
        verbose : bool, optional
            Whether to plot a progress bar (default) or not.

        Attributes
        ----------
        nside : int
            The healpix nside parameter, see a `healpy` manual
        ipix : aray of int, optional
            Healpy indices. If `None`, all sky is considered (default)
        l, b : array of float, shape is (len(ipix),)
            Galactic coordinates of pixel centres
        Omega : float
            Pixel areas
        D_ : array of float
            Edges of the distance grid
        D : array of float
            Centers of the distance grid
        dD : array of float
            Steps of the distance grid
        xyz : array of float, shape is (3, len(ipix), len(D))
            Galactocentric coordinates of the centres of the distance grid
            along the (l,b) positions
        dVol :
        """

        ipix = dom.ipix
        l = dom.l * u.deg
        b = dom.b * u.deg

        D = dom.D

        xyz = np.empty((3, len(ipix), len(D)))

        if verbose:
            iterator = tqdm(zip(l, b), total=len(ipix))
        else:
            iterator = zip(l, b)

        for i, lb_ in enumerate(iterator):

            # Transform coordinates from Galactic to Galactocentric
            l_, b_ = lb_
            co_gal  = SkyCoord(l_, b_, distance=D*u.kpc, frame='galactic')
            co_galc = co_gal.transform_to(Galactocentric)

            # Rays
            xyz[0,i,:] = co_galc.x.value
            xyz[1,i,:] = co_galc.y.value
            xyz[2,i,:] = co_galc.z.value

        self.xyz = xyz

        return self


    def load(self, fpath):
        d = np.load(fpath)
        self.__dict__.update(d)

        return self


    def save(self, fpath):
        np.savez_compressed(fpath, **self.__dict__)
