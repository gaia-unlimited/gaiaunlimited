
Citations and attributions
==========================

.. TODO: add links to the article
.. TODO: make use of __bibtex__

If you use this package, please cite Castro-Ginard et al. 2022 (in prep).

If you make use of empirical DR3 survey selection function
:py:class:`gaiaunlimited.selectionfunctions.DR3SelectionFunctionTCG`,
please cite `Cantat-Gaudin et al. 2022 <https://arxiv.org/abs/2208.09335>`_.


Data ported from Gaiaverse
------------------------------------------------

GaiaUnlimited builds on the previous work done by the `Completeness of Gaia-verse <https://www.gaiaverse.space>`_ project. Some of the selection functions and the improved scanning law for DR2 has been ported from that project. If you use results of their work, please cite the appropriate publication:

+ version='dr2_cog3' of :py:class:`gaiaunlimited.scanninglaw.GaiaScanningLaw`:
  `Boubert et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2954B/abstract>`_.
+ Survey selection functions for DR2:
  `Bouber & Everall 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4246B>`_.

    * :py:class:`gaiaunlimited.selectionfunctions.DR2SelectionFunction`

+ Sample selection functions from EDR3:
  `Everall & Boubert 2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.6205E>`_.

    * :py:class:`gaiaunlimited.selectionfunctions.DR3SelectionFunction`
