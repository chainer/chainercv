Installation Guide
==================

Pip
~~~

You can install ChainerCV using `pip`.

.. code-block:: shell

    pip install -U numpy
    pip install chainercv


Anaconda
~~~~~~~~

Build instruction using Anaconda is as follows.

.. code-block:: shell

    # For python 3
    # wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
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

