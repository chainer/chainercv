=========
ChainerCV
=========


ChainerCV is a **deep learning based computer vision library** built on top of `Chainer <https://github.com/chainer/chainer/>`_. 


Install Guide
=============

Pip
~~~

You can install ChainerCV using `pip`.

.. code-block:: shell

    # If Cython has not been installed yet, install it by a command like
    # pip install Cython
    pip install chainercv


Anaconda
~~~~~~~~

You can setup ChainerCV including its dependencies using anaconda.

.. code-block:: shell

    # For python 3
    # wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh

    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    
    # Download ChainerCV and go to the root directory of ChainerCV
    git clone https://github.com/chainer/chainercv
    cd chainercv
    conda env create -f environment.yml
    source activate chainercv

    # Install ChainerCV
    pip install -e .

    # Try our demos at examples/* !


Reference Manual
================

.. toctree::
   :maxdepth: 3

   reference/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
