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
+ Extensive worked examples of constructing selection functions for specialized samples crated form combinations of Gaia and other surveys.
+ Community contributed examples and tutorials.

Acknowledgements
----------------

This work is part of the GaiaUnlimited project funded by the European Union’s
Horizon 2020 research and innovation program under grant agreement No 101004110.

Team members
------------

Main contributors
`````````````````

Tristan Cantat-Gaudin (MPIA Heidelberg), 
Alfred Castro-Ginard (Leiden), 
Evgeny Kurbatov (IoA Cambridge), 
Shourya Khanna (INAF-Torino), 
Semyeong Oh (IoA Cambridge),

Members
```````

Anthony Brown (Leiden, PI), 
Vasily Belokurov (IoA Cambridge), 
Andy Casey (Monash), 
Alex Wallace (Monash), 
Ron Drimmel (INAF-Torino), 
Alessandro Spgna (INAF-Torino), 
Richard Smart (INAF-Torino), 
Eloisa Poggio (INAF-Torino), 
Morgan Fouesneau (MPIA Heidelberg), 
David Hogg (NYU/ Flatiron/ MPIA Heidelberg), 
Adrian Price-Whelan (Flatiron), 
Hans-Walter Rix (MPIA Heidelberg), 
Elena Sellentin (Leiden), 
Douglas Boubert (Oxford), 
Andy Everall (IoA Cambridge), 


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   notebooks/getting-started
   notebooks/dr3-empirical-completeness
   notebooks/constructing_SelectionFunctions
   notebooks/dr3-unresolved-binaries
   notebooks/scanninglaw
   citation
   api



.. toctree::
   :maxdepth: 1
   :caption: Tutorials and illustrated examples:

   notebooks/SubsampleSF_Tutorial.ipynb
   notebooks/SubsampleSF_HMLE_Tutorial.ipynb
   notebooks/rave_ratios.ipynb
   notebooks/APOGEE_selection_function.ipynb
   notebooks/RedClump_GSP-Spec_SF.ipynb
   notebooks/Aurora/README.md
   notebooks/RedClumpGaiaAllWISE/README.md
   notebooks/cube_galaxy_sky.ipynb
   community-examples

.. _Gaia: https://www.cosmos.esa.int/web/gaia/home
