import h5py
import healpy as hp
import numpy as np

from .. import fetch_utils, utils

__all__ = ["DR3SelectionFunctionForwardModel"]


class DR3SelectionFunctionForwardModel(fetch_utils.DownloadMixin):
    """Forward model of the Gaia DR3 survey selection function.

    Gives the probability that a source with a given sky position and G
    magnitude is included in Gaia DR3. The model is read directly from a
    precomputed table indexed by HEALPix pixel (order 7, nested, ICRS) and
    G-magnitude bin (0.2 mag wide, 1 < G < 22). The table is downloaded from
    Zenodo (https://doi.org/10.5281/zenodo.24190089) on first use.

    The class also exposes the recovered effective scanning law per HEALPix
    pixel: ``N_eff`` (posterior median of the effective number of scans),
    ``N_usable`` (number of usable scans) and ``N_error`` (posterior width of
    ``N_eff``, i.e. the difference between its 84th and 16th percentiles).

    Notes:
        Magnitude bins are 0.2 mag wide and closed on the right, (G_lo, G_hi];
        a magnitude on a bin edge (e.g. G = 21.0) falls in the brighter bin,
        (20.8, 21.0].

        :meth:`query` returns NaN where the model gives no estimate: (pixel, G)
        cells without Gaia sources, which are mostly at the bright end
        (G < 16), and any G outside the tabulated range [1, 22].

        The selection function overestimates the completeness of point sources
        at G > 21, so it should not be used there (see Sect. 5.3 of the paper).

    If you use this model in a publication please cite::

        @ARTICLE{2026arXiv261032551C,
                    author = {{Castro-Ginard}, Alfred and {Brown}, Anthony G.~A. and {Belotto Evangelista}, Eduardo and {Portell}, Jordi and {Casey}, Andrew R. and {Drimmel}, Ronald and {Sellentin}, Elena and {Cantat-Gaudin}, Tristan and {Fouesneau}, Morgan and {Khanna}, Shourya and et al.},
                    title = "{Forward model of the Gaia DR3 selection function}",
                    journal = {arXiv e-prints},
                    keywords = {Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
                    year = 2026,
                    month = oct,
                    eid = {arXiv:2610.32551},
                    pages = {arXiv:2610.32551},
                    doi = {10.48550/arXiv.2610.32551},
                    archivePrefix = {arXiv},
                    eprint = {2610.32551},
                    primaryClass = {astro-ph.GA},
                    adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv261032551C},
                    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }


    """

    __bibtex__ = """
        @ARTICLE{2026arXiv261032551C,
                    author = {{Castro-Ginard}, Alfred and {Brown}, Anthony G.~A. and {Belotto Evangelista}, Eduardo and {Portell}, Jordi and {Casey}, Andrew R. and {Drimmel}, Ronald and {Sellentin}, Elena and {Cantat-Gaudin}, Tristan and {Fouesneau}, Morgan and {Khanna}, Shourya and et al.},
                    title = "{Forward model of the Gaia DR3 selection function}",
                    journal = {arXiv e-prints},
                    keywords = {Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
                    year = 2026,
                    month = oct,
                    eid = {arXiv:2610.32551},
                    pages = {arXiv:2610.32551},
                    doi = {10.48550/arXiv.2610.32551},
                    archivePrefix = {arXiv},
                    eprint = {2610.32551},
                    primaryClass = {astro-ph.GA},
                    adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv261032551C},
                    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }
"""

    datafiles = {
        "dr3-forward-model-hpx7.h5": "https://zenodo.org/records/24190089/files/dr3-forward-model-hpx7.h5",
    }

    def __init__(self):
        with h5py.File(self._get_data("dr3-forward-model-hpx7.h5"), "r") as f:
            self.healpix_order = int(f.attrs["healpix_order"])
            self.g_centers = f["g_centers"][()]
            # Clip round-off (e.g. -1e-11) to [0, 1]; NaNs are kept.
            self.p = np.clip(f["p"][()], 0.0, 1.0)  # shape (npix, n_gbins)
            self.N_eff = f["N_true"][()]  # stored as "N_true" in the Zenodo file
            self.N_usable = f["N_usable"][()]
            self.N_error = f["N_error"][()]
        self.nside = hp.order2nside(self.healpix_order)

        half_width = 0.5 * (self.g_centers[1] - self.g_centers[0])
        # round so that edges are exact (e.g. 20.8, not 20.799999999999997)
        self.g_edges = np.round(
            np.append(self.g_centers - half_width, self.g_centers[-1] + half_width), 6
        )

    def _pixels(self, coords):
        return utils.coord2healpix(coords, "icrs", self.nside, nest=True)

    def query(self, coords, gmag):
        """Query the selection function.

        Args:
            coords: sky coordinates as an astropy coordinates instance.
            gmag (float or array): Gaia G magnitudes. Should have the same shape as coords.

        Returns:
            np.array: selection probabilities, same shape as coords. NaN where the
            model gives no estimate or where G is outside [1, 22].
        """
        gmag = np.asarray(gmag, dtype=float)
        if coords.shape != gmag.shape:
            raise ValueError(f"Input shape mismatch: {coords.shape} != {gmag.shape}")

        ipix = np.atleast_1d(self._pixels(coords))
        g = np.atleast_1d(gmag)

        in_range = (g >= self.g_edges[0]) & (g <= self.g_edges[-1])
        # bins are (G_lo, G_hi]; G = 1 exactly goes to the first bin
        igbin = np.clip(np.searchsorted(self.g_edges, g, side="left") - 1, 0, len(self.g_centers) - 1)

        prob = np.full(g.shape, np.nan)
        prob[in_range] = self.p[ipix[in_range], igbin[in_range]]

        if coords.ndim == 0:
            return prob.squeeze()
        return prob

    def query_effective_scanning_law(self, coords):
        """Query the recovered effective scanning law at given sky positions.

        Args:
            coords: sky coordinates as an astropy coordinates instance.

        Returns:
            dict: keys ``"N_eff"``, ``"N_usable"`` and ``"N_error"`` (posterior
            width of ``N_eff``), each an array with the same shape as coords.
        """
        ipix = self._pixels(coords)
        return {
            "N_eff": self.N_eff[ipix],
            "N_usable": self.N_usable[ipix],
            "N_error": self.N_error[ipix],
        }
