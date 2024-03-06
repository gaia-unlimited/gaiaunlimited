Installation
------------

The package is available through PyPI and Github.

To install gaiaunlimited with pip, run:

.. code-block:: bash

   pip install gaiaunlimited

To install gaiaunlimited through github, run:

.. code-block:: bash

   pip install gaiaunlimited@git+https://github.com/gaia-unlimited/gaiaunlimited@<latest release>

Alternatively, clone the repository and install:

.. code-block:: bash

   git clone https://github.com/gaia-unlimited/gaiaunlimited.git
   cd gaiaunlimited
   python -m pip install .

In case a local installation is not possible, the package can also be installed inside a Google Colab notebook by executing the command ``!pip install gaiaunlimited`` or  ``%pip install gaiaunlimited``.

Optional: setting the data directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaiaunlimited downloads and caches large binary data files. To set the data directory to store these files,
set the environment variable ``GAIAUNLIMITED_DATADIR`` to the desired location. By default, this will be ``~/.gaiaunlimited``.

.. code-block:: bash

    export GAIAUNLIMITED_DATADIR="/path/to/directory"
