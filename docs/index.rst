.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the gaiaunlimited documentation!
===========================================

`gaiaunlimited` is a Python package for querying and constructing selection functions for the `Gaia
<https://www.cosmos.esa.int/web/gaia/home>`_ survey. It is developed by the `GaiaUnlimited collaboration
<https://gaia-unlimited.org>`_.

The GaiaUnlimited project is funded by the European Union’s Horizon 2020 research and
innovation program under grant agreement No 101004110.

.. attention::

   This package is being actively developed in `a public repository on GitHub <http://github.com/gaia-unlimited/gaiaunlimited>`_, and we
   need your input!
   If you have any trouble with this code, find a typo, or have requests for new
   content (tutorials or features), please `open an issue on GitHub <https://github.com/gaia-unlimited/gaiaunlimited/issues/new/choose>`_.


Features
--------

+ Query Gaia scanning laws for when a given sky position is scanned.
+ Query Gaia survey selection functions for the probabilities that sources of given properties are included in the Gaia catalog.
+ Query ready-made Gaia DR3 subsample selection functions for the astrometry and RVS sample.

Acknowledgements
----------------

This work is part of the GaiaUnlimited project funded by the European Union’s
Horizon 2020 research and innovation program under grant agreement No 101004110.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   scanninglaw
   getting-started
   dr3-empirical-completeness
   dr3-rvs
   citation
   api

.. _Gaia: https://www.cosmos.esa.int/web/gaia/home
