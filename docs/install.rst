Installation
------------

The package is available through PyPI and Github.

To install gaiasf with pip, run:

.. code-block:: bash

   pip install gaiasf

To install gaiasf through github, run:

.. code-block:: bash

   pip install gaiasf@git+https://github.com/gaia-unlimited/gaiasf@<latest release>

Setting the data directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaiasf downloads and caches large binary data files. To set the data directory to store these files,
set the environment variable ``GAIASF_DATADIR`` to the desired location. By default, this will be ``~/.gaiasf``.

.. code-block:: bash

    export GAIASF_DATADIR="/path/to/directory"
