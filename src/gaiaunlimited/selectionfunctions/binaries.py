import numpy as np
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle

from .. import fetch_utils, utils

class BinarySystemsSelectionFunction(fetch_utils.DownloadMixin):
    """Model to estimate detectability of binary systems in Gaia DR3 based on RUWE.

    If you use this model in a publication please cite (change for new paper):

        }
    """
    
    datafiles = {
        "dict_SL_ruwe.pkl": " "#"https://zenodo.org/record/8063930/files/allsky_M10_hpx7.hdf5"
   }

    def __init__(self):
        print("WARNING: This functionality is currently under development. Use with caution.")
#############################################################################################
        try:
            with open(self._get_data("dict_SL_ruwe.pkl"),'rb') as f:
                SL_hpx5 = pickle.load(f)
        except:
            print("WARNING: missing data file.")
            raise
#############################################################################################
        self.ra = np.hstack([SL_hpx5[i]['ra_degrees'] for i in range(hp.order2npix(5))])
        self.dec = np.hstack([SL_hpx5[i]['dec_degrees'] for i in range(hp.order2npix(5))])
        self.ruwe_threshold = np.hstack([SL_hpx5[i]['ruwe_threshold'] for i in range(hp.order2npix(5))])
        self.ruwe_simulation = np.hstack([SL_hpx5[i]['ruwe_simulation'] for i in range(hp.order2npix(5))])
        self.selection_probabilities = np.hstack([SL_hpx5[i]['selection_probabilities'] for i in range(hp.order2npix(5))])
        self.selection_probability_variances = np.hstack([SL_hpx5[i]['selection_probability_variances'] for i in range(hp.order2npix(5))])
        self.observation_times = np.array([SL_hpx5[i]['observation_times_years'] for i in range(hp.order2npix(5))],dtype = object)
        self.scanning_angles = np.array([SL_hpx5[i]['scanning_angles_radians'] for i in range(hp.order2npix(5))],dtype = object)
        self.AL_parallax_factors = np.array([SL_hpx5[i]['AL_parallax_factor'] for i in range(hp.order2npix(5))],dtype = object)
        
    def query(self,coords,return_variance = False):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            return_variance: bool, return variance on the selection probabilities. Default = False.
        Returns:
            prob: array of selection probabilities.
        """
        if not isinstance(coords, SkyCoord):
            print(
                "*** WARNING: The coordinates do not seem to be an astropy.coord.SkyCoord object."
            )
            print(
                "This could lead to the error:\n     \"object has no attribute 'frame'\""
            )
            print(
                "The syntax to query the completeness map is:\n    mapName.query( coordinates )"
            )
        if coords.frame.name != 'icrs':
            coords = coords.transform_to('icrs')
        ra = coords.ra.deg
        dec = coords.dec.deg
        hp_index = hp.ang2pix(hp.order2nside(5),ra,dec,lonlat = True,nest = False)
        if return_variance:
            return self.selection_probabilities[hp_index],self.selection_probability_variances[hp_index]
        else:
            return self.selection_probabilities[hp_index]
        
    def query_ScanningLaw(self,coords):
        """Query the scanning law parameters.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
        Returns:
            observation_times: array of observation times in years.
            scanning_angles: array of scanning angles in radians.
            AL_parallax_factors: array of along scan parallax factors.
        """
        if not isinstance(coords, SkyCoord):
            print(
                "*** WARNING: The coordinates do not seem to be an astropy.coord.SkyCoord object."
            )
            print(
                "This could lead to the error:\n     \"object has no attribute 'frame'\""
            )
            print(
                "The syntax to query the completeness map is:\n    mapName.query( coordinates )"
            )
        if coords.frame.name != 'icrs':
            coords = coords.transform_to('icrs')
        ra = coords.ra.deg
        dec = coords.dec.deg
        hp_index = hp.ang2pix(hp.order2nside(5),ra,dec,lonlat = True,nest = False)
        return self.observation_times[hp_index],self.scanning_angles[hp_index],self.AL_parallax_factors[hp_index]
    
    def query_RUWE(self,coords,crowding = True):
        """Query the estimated RUWE threshold.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            crowding: bool. Include offset estimated from crowding. Default = True.
        Returns:
            ruwe_threshold: array of RUWE threshold including crowding effects.
            ruwe_simulation: array of RUWE threshold based on simulated single sources.
        """
        if not isinstance(coords, SkyCoord):
            print(
                "*** WARNING: The coordinates do not seem to be an astropy.coord.SkyCoord object."
            )
            print(
                "This could lead to the error:\n     \"object has no attribute 'frame'\""
            )
            print(
                "The syntax to query the completeness map is:\n    mapName.query( coordinates )"
            )
        if coords.frame.name != 'icrs':
            coords = coords.transform_to('icrs')
        ra = coords.ra.deg
        dec = coords.dec.deg
        hp_index = hp.ang2pix(hp.order2nside(5),ra,dec,lonlat = True,nest = False)
        if crowding:
            return self.ruwe_threshold[hp_index]
        else:
            return self.ruwe_simulation[hp_index]