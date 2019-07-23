# Anamic

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/brouhardlab/anamic/master?urlpath=lab/tree/notebooks)
[![Build Status](https://travis-ci.com/brouhardlab/anamic.svg?branch=master)](https://travis-ci.com/brouhardlab/anamic)

Simulate, fit and analyze microtubules.

![drawing](doc/drawing.png)

## Dependencies

- python > 3.6 (stop using Python 2!)

## Install

Clone the repository:

```bash
git clone https://github.com/brouhardlab/anamic.git
```

We suggest you to install Python from the [Anaconda distribution](https://www.anaconda.com/distribution/). Then you can install the required dependencies in a Conda environment:

```bash
cd anamic/
conda create -n anamic_env
conda activate anamic_env
conda env update -f binder/environment.yml
```

Note that anamic is also a valid Python library with `pip`:

```bash
pip install anamic
# or
pip install https://github.com/brouhardlab/anamic/archive/master.zip
```

## License

Under BSD license. See [LICENSE](LICENSE).

## Authors

- Claire Edrington <>
- Hadrien Mary <hadrien.mary@gmail.com>

## Release a new version

- Run tests: `pytest -v anamic/`.
- Install [rever](https://regro.github.io/rever-docs): `conda install -y rever`.
- Run check: `rever check`.
- Bump and release new version: `rever VERSION_NUMBER`.
