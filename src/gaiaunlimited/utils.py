import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np

from typing import Iterable, Optional, Tuple, Union
from astropy import constants, coordinates
import pickle
from astromet import sigma_ast
import time

from . import fetch_utils

__all__ = ["coord2healpix", "get_healpix_centers","SimulateGaiaSource"]


def get_healpix_centers(order,nest=False):
    """Get centers of HEALPix as astropy coordinates.

    Args:
        order (int): The order of the pixelisation

    Optional args:
        nest (boolean): ``False`` for RING healpix scheme, ``True`` for nested.
            default is ``False``.

    Returns:
        astropy.coordinates.SkyCoord: coordinates of the centers.
    """
    nside = hp.order2nside(order)
    npix = hp.order2npix(order)
    ipix = np.arange(npix)
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True, nest=nest)
    return coord.SkyCoord(ra * u.deg, dec * u.deg)


def coord2healpix(coords, frame, nside, nest=True):
    """
    Calculate HEALPix indices from an astropy SkyCoord. Assume the HEALPix
    system is defined on the coordinate frame ``frame``.

    Args:
        coords (astropy.coordinates.BaseCoordinateFrame): The input coordinates.
            This can be SkyCoord for any other specific frame instance that can
            be converted to a SkyCoord.
        frame (str): The frame in which the HEALPix system is defined. Should be
            a valid name from astropy.coordinates.
        nside (int): The HEALPix nside parameter to use. Must be a power of 2.
        nest (bool): True if the HEALPix uses nested scheme.

    Returns:
        An array of pixel indices with the same shape as the input
        coords.

    Raises:
        ValueError: The specified frame is not supported.
    """
    # In case coords is specific frame objects, make it SkyCoord.
    # Otherwise, there is no 'frame' attribute.
    if isinstance(coords, coord.SkyCoord):
        coords = coord.SkyCoord(coords)
    if coords.frame.name != frame:
        c = coords.transform_to(frame)
    else:
        c = coords

    if hasattr(c, "ra"):
        phi = c.ra.rad
        theta = 0.5 * np.pi - c.dec.rad
        return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    elif hasattr(c, "l"):
        phi = c.l.rad
        theta = 0.5 * np.pi - c.b.rad
        return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    elif hasattr(c, "x"):
        return hp.pixelfunc.vec2pix(nside, c.x.kpc, c.y.kpc, c.z.kpc, nest=nest)
    elif hasattr(c, "w"):
        return hp.pixelfunc.vec2pix(nside, c.w.kpc, c.u.kpc, c.v.kpc, nest=nest)
    else:
        raise ValueError(
            'No method to transform from coordinate frame "{}" to HEALPix.'.format(
                frame
            )
        )

class SimulateGaiaSource(fetch_utils.DownloadMixin):

    """Forward model to estimate RUWE for single sources or binary systems in Gaia DR3.

    If you use this model in a publication please cite:

    @ARTICLE{2024arXiv240414127C,
       author = {{Castro-Ginard}, Alfred and {Penoyre}, Zephyr and {Casey}, Andrew R. and {Brown}, Anthony G.~A. and {Belokurov}, Vasily and {Cantat-Gaudin}, Tristan and {Drimmel}, Ronald and {Fouesneau}, Morgan and {Khanna}, Shourya and {Kurbatov}, Evgeny P. and {Price-Whelan}, Adrian M. and {Rix}, Hans-Walter and {Smart}, Richard L.},
        title = "{Gaia DR3 detectability of unresolved binary systems}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = apr,
          eid = {arXiv:2404.14127},
        pages = {arXiv:2404.14127},
          doi = {10.48550/arXiv.2404.14127},
archivePrefix = {arXiv},
       eprint = {2404.14127},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240414127C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

        }
    """

    datafiles = {
        "dict_SL_ruwe.pkl": "https://zenodo.org/records/11102437/files/dict_SL_ruwe.pkl"
   }
    
    def __init__(self, ra, dec, period=0, eccentricity=0, initial_phase=0, epoch=2016.0):
        
        with open(self._get_data("dict_SL_ruwe.pkl"),'rb') as f:
            self.SL_hpx5 = pickle.load(f)
        self.ra = ra
        self.dec = dec
        self.epoch = epoch
        coord = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        phi_ = coord.ra.deg
        theta_ = coord.dec.deg
        self.hp_ind = hp.ang2pix(hp.order2nside(5),phi_,theta_,lonlat = True,nest = False)
        self.t_obs = np.array(self.SL_hpx5[self.hp_ind]['observation_times_years'])
        self.scan_angle = np.array(self.SL_hpx5[self.hp_ind]['scanning_angles_radians'])
        self.parallax_factor = np.array(self.SL_hpx5[self.hp_ind]['AL_parallax_factor'])
        self.set_period_eccentricity_and_phase(period, eccentricity, initial_phase)
        return None
    def set_period_eccentricity_and_phase(self, period, eccentricity, initial_phase):
        self.period = period
        self.eccentricity = eccentricity
        self.initial_phase = initial_phase
        self._calculate_eta_and_phi()
        return None
    def _calculate_eta_and_phi(self):
        self.η = np.atleast_2d(
            eta(self.t_obs, self.period, self.eccentricity, self.initial_phase)
        )
        self.ϕ = (
            2 * np.arctan(np.sqrt((1 + self.eccentricity)/(1 - self.eccentricity)) \
            * np.tan(0.5 * self.η)) % (2 * np.pi)
        ).T
        self.cos_ϕ = np.cos(self.ϕ)
        self.sin_ϕ = np.sin(self.ϕ)
        self.a_to_r = (1 - self.eccentricity * np.cos(self.η)).T
        return None
    def observe(self, phot_g_mean_mag, parallax, a, q, l, phi, theta, omega):
        phot_g_mean_mag, parallax, a, q, l, phi, theta, omega = map(
            np.atleast_2d, 
            (phot_g_mean_mag, parallax, a, q, l, phi, theta, omega)
        )
        r = self.a_to_r @ a
        g = (1-(np.cos(phi)**2)*(np.sin(theta)**2))**-0.5
        x_com = r*g*(self.cos_ϕ - np.cos(phi - self.ϕ)*np.cos(phi)*(np.sin(theta)**2))
        y_com = r*g*self.sin_ϕ*np.cos(theta)
        inv_1pq = 1/(1+q)
        ratio = abs(l-q)/((1+l)*(1+q)) # -1
        x_p = x_com * ratio
        y_p = y_com * ratio
        ra_p = parallax*(x_p*np.cos(omega) + y_p*np.sin(omega))
        de_p = parallax*(y_p*np.cos(omega) - x_p*np.sin(omega))
        al_positions = ra_p.T * np.sin(self.scan_angle) + de_p.T * np.cos(self.scan_angle)
        al_errors = sigma_ast(phot_g_mean_mag).flatten()
        al_positions += np.random.normal(np.zeros_like(al_positions),al_errors.reshape((-1,1)))
        return (al_positions, al_errors)
    def unit_weight_error(self, al_positions, al_errors,crowding = False):
        N, T = al_positions.shape
        assert al_errors.size == N
        A = self.design_matrix_solveRUWE
        C = np.linalg.solve(A.T @ A, np.eye(5))
        ACAT = A @ C @ A.T
        R = al_positions - al_positions @ ACAT
        if crowding: 
            offset = np.array(self.SL_hpx5[self.hp_ind]['ruwe_threshold']) - np.array(self.SL_hpx5[self.hp_ind]['ruwe_simulation'])
            return np.sqrt(np.sum((R/al_errors.reshape((-1, 1)))**2, axis=1) / (T - 5)) + offset
        else:
            return np.sqrt(np.sum((R/al_errors.reshape((-1, 1)))**2, axis=1) / (T - 5))
    @property
    def design_matrix_solveRUWE(self):
        return np.column_stack([
            np.sin(self.scan_angle),
            np.cos(self.scan_angle),
            self.parallax_factor,
            (self.t_obs - self.epoch) * np.sin(self.scan_angle),
            (self.t_obs - self.epoch) * np.cos(self.scan_angle),
        ])
    
def eta(
    t: Iterable[float], 
    period: float, 
    eccentricity: float, 
    initial_phase: float, 
    max_iter: Optional[int] = 30
) -> Iterable[float]:
    """
    Calculate the true anomaly at the given times, given the orbital parameters.
    
    :param t:
        The time of observation(s) [jyear].
    
    :param period:
        The orbital period of the binary [years].
    
    :param eccentricity:
        The eccentricity of the binary orbit.
    
    :param initial_phase:
        The phase of the orbit at `t = 0` [radians].
    
    :param max_iter: [optional]
        The maximum number of iterations to perform when solving for the true anomaly.
    
    :returns:
        The true anomaly at the given times [radians].
    """
    if period == 0:
        return np.zeros_like(t)
    
    ω = (2 * np.pi * (t / period) + initial_phase) % (2 * np.pi)
    sin_ω = np.sin(ω)
    cos_ω = np.cos(ω)
    η = (
        ω 
      + eccentricity*sin_ω 
      + (eccentricity**2)*sin_ω*cos_ω 
      + 0.5*(eccentricity**3)*sin_ω*(3*(cos_ω**2)-1)
    )

    for n in range(max_iter):
        sin_η = np.sin(η)
        cos_η = np.cos(η)
        f = η - eccentricity*sin_η - ω
        d_f = 1 - eccentricity*cos_η
        d2_f2 = eccentricity*sin_η
        delta_η = -f*d_f / (d_f*d_f - 0.5*f*d2_f2)
        η += delta_η
        if (np.max(np.abs(delta_η)) < 1e-5):
            break
    #else:
    #    logger.warn(f"Did not converge after {max_iter} iterations ({delta_η:.1e}).")
    
    return η

earth_sun_mass_ratio = (constants.M_earth/constants.M_sun).value

def lagrangian_point_2_coordinates(t):
    pos = coordinates.get_body_barycentric('earth', time.Time(t, format='jyear'))
    l2corr = 1 + (earth_sun_mass_ratio/3)**(1/3)
    return l2corr * np.vstack([pos.x.value, pos.y.value, pos.z.value]).T

