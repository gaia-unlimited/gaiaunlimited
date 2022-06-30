import numpy as np
import healpy as hp

__all__ = ['coord2healpix']

def coord2healpix(coords, frame, nside, nest=True):
    """
    Calculate HEALPix indices from an astropy SkyCoord. Assume the HEALPix
    system is defined on the coordinate frame ``frame``.

    Args:
        coords (:obj:`astropy.coordinates.SkyCoord`): The input coordinates.
        frame (:obj:`str`): The frame in which the HEALPix system is defined.
        nside (:obj:`int`): The HEALPix nside parameter to use. Must be a power of 2.
        nest (Optional[:obj:`bool`]): ``True`` (the default) if nested HEALPix ordering
            is desired. ``False`` for ring ordering.

    Returns:
        An array of pixel indices (integers), with the same shape as the input
        SkyCoord coordinates (:obj:`coords.shape`).

    Raises:
        :obj:`sfexceptions.CoordFrameError`: If the specified frame is not supported.
    """
    if coords.frame.name != frame:
        c = coords.transform_to(frame)
    else:
        c = coords

    if hasattr(c, 'ra'):
        phi = c.ra.rad
        theta = 0.5*np.pi - c.dec.rad
        return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    elif hasattr(c, 'l'):
        phi = c.l.rad
        theta = 0.5*np.pi - c.b.rad
        return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    elif hasattr(c, 'x'):
        return hp.pixelfunc.vec2pix(nside, c.x.kpc, c.y.kpc, c.z.kpc, nest=nest)
    elif hasattr(c, 'w'):
        return hp.pixelfunc.vec2pix(nside, c.w.kpc, c.u.kpc, c.v.kpc, nest=nest)
    else:
        raise ValueError(
            'No method to transform from coordinate frame "{}" to HEALPix.'.format(
                frame))