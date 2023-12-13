import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import numpy as np

__all__ = ["coord2healpix", "get_healpix_centers"]


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
