# %%

import numpy as np

from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord



class Domain():

    def __init__(self):
        pass


    def make(self, nside, Dlim, Dnum, radius=None):

        #
        # Sky

        self.nside = nside

        hp = HEALPix(nside=nside, order='nested', frame='icrs')

        # Pixel indices
        if radius is None:
            hpx_order = int(np.log2(nside))
            npix = 12 * 4**hpx_order
            self.ipix = np.arange(npix)
        else:
            co = SkyCoord(0.0, 0.0, frame='galactic', unit='deg')
            self.ipix = hp.cone_search_skycoord(co, radius=radius)

        # Pixel centres
        co = hp.healpix_to_skycoord(self.ipix)
        self.l = co.galactic.l.value
        self.b = co.galactic.b.value
        # Pixel areas
        self.Omega = hp.pixel_area.value

        #
        # Distances

        # Edges of the distance grid
        self.D_ = np.linspace(Dlim[0], Dlim[1], Dnum)
        # Centers of the distance grid
        self.D = 0.5*(self.D_[:-1] + self.D_[1:])
        # Steps of the distance grid
        self.dD = self.D_[1:] - self.D_[:-1]

        #
        # Measures

        self.dVol = self.Omega * (self.D_[1:]**3 - self.D_[:-1]**3) / 3.0

        return self


    def load(self, fpath):
        d = np.load(fpath)
        self.__dict__.update(d)

        return self


    def save(self, fpath):
        np.savez_compressed(fpath, **self.__dict__)
